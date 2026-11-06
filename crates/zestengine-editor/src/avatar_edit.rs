//! Editor helpers for avatar authoring: finding the avatar around an object,
//! adding a toggle, the synced-bits total shown in the inspectors, and the
//! live preview of a toggle (its actions applied to the scene until stopped,
//! then restored exactly).

use zestengine_core::{
    AvatarDescriptor, AvatarParameters, BlendShapes, MenuToggle, ObjectId, ToggleAction, ToggleMode, component_enabled, set_component_enabled,
};

use zestengine_engine::scene::LoadedScene;

/// The avatar root at or above `i`: the nearest object with an Avatar Descriptor.
pub fn avatar_root_of(scene: &LoadedScene, i: usize) -> Option<usize> {
    std::iter::successors(Some(i), |&k| scene.objects[k].parent)
        .take(256)
        .find(|&k| scene.objects[k].components.iter().any(|c| c.as_any().is::<AvatarDescriptor>()))
}

/// Path of object `i` below `root` ("" for the root itself).
pub fn path_below(scene: &LoadedScene, root: usize, i: usize) -> Option<String> {
    let mut names = Vec::new();
    let mut k = i;
    for _ in 0..256 {
        if k == root {
            names.reverse();
            return Some(names.join("/"));
        }
        names.push(scene.objects[k].name.as_str());
        k = scene.objects[k].parent?;
    }
    None
}

/// The object at `path` below `root`.
pub fn resolve_below(scene: &LoadedScene, root: usize, path: &str) -> Option<usize> {
    let mut at = root;
    for seg in path.split('/').filter(|s| !s.is_empty()) {
        at = (0..scene.objects.len()).find(|&k| scene.objects[k].parent == Some(at) && scene.objects[k].name == seg)?;
    }
    Some(at)
}

/// Add a toggle that shows or hides object `i`: a `Menu Toggle` on it with an
/// object action, and the object hidden (the resting look). Returns a message
/// when it cannot.
pub fn add_toggle(scene: &mut LoadedScene, i: usize) -> Result<(), &'static str> {
    let root = avatar_root_of(scene, i).ok_or("Add Toggle needs an object under an Avatar Descriptor")?;
    if root == i {
        return Err("Pick an object below the avatar root to toggle");
    }
    let path = path_below(scene, root, i).ok_or("Could not work out the object's path")?;
    let name = scene.objects[i].name.clone();
    let toggle = MenuToggle {
        menu_path: name,
        actions: vec![ToggleAction::Active { target: path, on: true }],
        ..Default::default()
    };
    scene.objects[i].components.push(Box::new(toggle));
    scene.objects[i].enabled = false;
    Ok(())
}

/// Synced bits of the avatar around `i`: declared parameters plus every
/// toggle's own parameter (a close estimate of what the build adds up to), and
/// the number of toggles. `None` when `i` is not under an avatar.
pub fn bit_totals(scene: &LoadedScene, i: usize) -> Option<(u32, u32, usize)> {
    let root = avatar_root_of(scene, i)?;
    let mut params: Vec<zestengine_core::AvatarParam> = Vec::new();
    let mut toggle_bits = 0;
    let mut toggles = 0;
    let mut selectors: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut max = zestengine_core::avatar_params::DEFAULT_PARAM_BUDGET_BITS;
    for k in 0..scene.objects.len() {
        if !std::iter::successors(Some(k), |&p| scene.objects[p].parent).take(256).any(|p| p == root) {
            continue;
        }
        for c in &scene.objects[k].components {
            if let Some(ap) = c.as_any().downcast_ref::<AvatarParameters>() {
                if k == root {
                    max = ap.budget_bits;
                }
                params.extend(ap.params.iter().cloned());
            } else if let Some(t) = c.as_any().downcast_ref::<MenuToggle>() {
                toggles += 1;
                toggle_bits += t.synced_bit_cost();
                // A selector group is one Int however many options it has.
                let sel = t.selector.trim();
                if t.synced && !sel.is_empty() && !t.mode.is_slider() && selectors.insert(sel.to_owned()) {
                    toggle_bits += zestengine_core::AvatarParamKind::Int.bits();
                }
            }
        }
    }
    Some((zestengine_core::synced_bits(&params) + toggle_bits, max, toggles))
}

// ------------------------------------------------------------------- preview

enum Saved {
    Enabled(ObjectId, bool),
    Component(ObjectId, String, bool),
    Blend(ObjectId, String, Option<f32>),
}

/// A toggle's actions applied to the scene, with what they replaced.
#[derive(Default)]
pub struct TogglePreview {
    source: Option<ObjectId>,
    saved: Vec<Saved>,
}

impl TogglePreview {
    /// The preview of the toggle on object `id` is applied.
    pub fn is_on(&self, id: ObjectId) -> bool {
        self.source == Some(id)
    }
    pub fn active(&self) -> bool {
        self.source.is_some()
    }

    /// Show every toggle on object `i` as on. Material actions are not
    /// previewed (they need the renderer); they show up in Play.
    pub fn start(&mut self, scene: &mut LoadedScene, i: usize) {
        self.restore(scene);
        let Some(root) = avatar_root_of(scene, i) else { return };
        let actions: Vec<ToggleAction> = scene.objects[i]
            .components
            .iter()
            .filter_map(|c| c.as_any().downcast_ref::<MenuToggle>())
            .flat_map(|t| {
                // A slider previews at full.
                let _ = matches!(t.mode, ToggleMode::Slider { .. });
                t.actions.clone()
            })
            .collect();
        self.source = Some(scene.objects[i].id);
        for a in actions {
            let Some(target) = a.target().and_then(|p| resolve_below(scene, root, p)) else { continue };
            let id = scene.objects[target].id;
            match a {
                ToggleAction::Active { on, .. } => {
                    self.saved.push(Saved::Enabled(id, scene.objects[target].enabled));
                    scene.objects[target].enabled = on;
                }
                ToggleAction::ComponentEnabled { component, on, .. } => {
                    if let Some(c) = scene.objects[target].components.iter_mut().find(|c| c.type_name() == component) {
                        self.saved.push(Saved::Component(id, component.clone(), component_enabled(c.as_ref())));
                        set_component_enabled(c, on);
                    }
                }
                ToggleAction::BlendShape { shape, value, .. } => {
                    let comps = &mut scene.objects[target].components;
                    if !comps.iter().any(|c| c.as_any().is::<BlendShapes>()) {
                        comps.push(Box::new(BlendShapes::default()));
                    }
                    if let Some(b) = comps.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<BlendShapes>()) {
                        let before = b.weights.iter().find(|(n, _)| *n == shape).map(|(_, w)| *w);
                        self.saved.push(Saved::Blend(id, shape.clone(), before));
                        b.set(&shape, value);
                    }
                }
                _ => {}
            }
        }
    }

    /// Put everything back exactly as it was.
    pub fn restore(&mut self, scene: &mut LoadedScene) {
        if self.source.is_none() {
            return;
        }
        let index = scene.id_index();
        for s in self.saved.drain(..).rev() {
            match s {
                Saved::Enabled(id, v) => {
                    if let Some(&k) = index.get(&id) {
                        scene.objects[k].enabled = v;
                    }
                }
                Saved::Component(id, name, v) => {
                    if let Some(&k) = index.get(&id)
                        && let Some(c) = scene.objects[k].components.iter_mut().find(|c| c.type_name() == name)
                    {
                        set_component_enabled(c, v);
                    }
                }
                Saved::Blend(id, shape, before) => {
                    if let Some(&k) = index.get(&id)
                        && let Some(b) = scene.objects[k].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<BlendShapes>())
                    {
                        match before {
                            Some(w) => b.set(&shape, w),
                            None => b.weights.retain(|(n, _)| *n != shape),
                        }
                    }
                }
            }
        }
        self.source = None;
    }
}
