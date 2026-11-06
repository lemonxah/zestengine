//! Runs the avatar build when Play starts.
//!
//! For each avatar in the scene that has authoring components, the scene is
//! serialised, the build runs on that copy (`avatar_build`, never moving
//! objects: the editor's Play snapshot only restores components and the
//! active flag), and the result is written back to the live objects. The
//! generated assets land in `.zest/generated/avatars/<id>/`. Stopping Play
//! restores the authored components from the snapshot taken before this ran,
//! so the authored avatar is never changed.

use std::path::Path;

use zestengine_assets::avatar_build::{self, BuildEnv};
use zestengine_assets::{ComponentData, SceneFile};
use zestengine_core::{Component, ComponentRegistry};

use crate::scene::LoadedScene;
use crate::shaders::ShaderLibrary;

/// What one avatar's build reported, for the status line and the log.
#[derive(Clone, Debug, Default)]
pub struct PlayBuildReport {
    pub avatar: String,
    pub layers: usize,
    pub generated_params: usize,
    pub synced_bits: u32,
    pub max_bits: u32,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

/// The components of an object rebuilt from `saved`, keeping untouched live
/// ones (and their runtime state) as they are.
fn merge_components(old: Vec<Box<dyn Component>>, saved: &[ComponentData], registry: &ComponentRegistry) -> Vec<Box<dyn Component>> {
    let mut old: Vec<Option<(ComponentData, Box<dyn Component>)>> = old.into_iter().map(|c| Some((ComponentData::of(c.as_ref()), c))).collect();
    saved
        .iter()
        .filter_map(|cd| {
            let same = old.iter_mut().find(|o| o.as_ref().is_some_and(|(d, _)| d == cd));
            match same.and_then(Option::take) {
                Some((_, c)) => Some(c),
                None => cd.load(registry),
            }
        })
        .collect()
}

/// Build every avatar in `scene` that needs it and apply the result.
pub fn build_for_play(scene: &mut LoadedScene, project_root: &Path, shaders: &ShaderLibrary, registry: &ComponentRegistry, budget_cap: u32) -> Vec<PlayBuildReport> {
    let mut out = Vec::new();
    // A cheap check first: serialising the scene is the expensive part.
    let any = scene.objects.iter().any(|o| {
        o.components.iter().any(|c| {
            let n = c.type_name();
            avatar_build::authoring_kinds().contains(&n) || n == "Avatar Descriptor"
        })
    });
    if !any {
        return out;
    }
    let file: SceneFile = scene.to_scene_file(project_root, shaders);
    for root in avatar_build::avatar_roots(&file) {
        if !avatar_build::needs_build(&file, root) {
            continue;
        }
        let env = BuildEnv { project_root, write: true, structural: false, budget_cap };
        let (built, report) = avatar_build::build_avatar_cloned(&file, root, &env);
        for &i in &avatar_build::subtree(&file, root) {
            let (Some(old), Some(new)) = (file.entries.get(i), built.entries.get(i)) else { continue };
            if old.enabled != new.enabled {
                scene.objects[i].enabled = new.enabled;
            }
            if old.components != new.components {
                let live = std::mem::take(&mut scene.objects[i].components);
                scene.objects[i].components = merge_components(live, &new.components, registry);
            }
        }
        for w in &report.warnings {
            tracing::warn!("avatar build ({}): {w}", file.entries[root].name);
        }
        for e in &report.errors {
            tracing::error!("avatar build ({}): {e}", file.entries[root].name);
        }
        out.push(PlayBuildReport {
            avatar: file.entries[root].name.clone(),
            layers: report.layers.len(),
            generated_params: report.generated_params.len(),
            synced_bits: report.budget.map_or(0, |b| b.used),
            max_bits: report.budget.map_or(0, |b| b.max),
            warnings: report.warnings,
            errors: report.errors,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};
    use zestengine_assets::ObjectSource;
    use zestengine_core::{Animator, AvatarDescriptor, AvatarMenuData, AvatarParameters, MenuToggle, ObjectId, ToggleAction};

    use crate::scene::SceneObject;

    fn obj(scene: &mut LoadedScene, name: &str, parent: Option<usize>, components: Vec<Box<dyn Component>>) -> usize {
        scene.objects.push(SceneObject {
            net_role: Default::default(),
            component_roles: Default::default(),
            id: ObjectId::new(),
            name: name.into(),
            render: None,
            extra_render: Vec::new(),
            source: ObjectSource::Empty,
            enabled: true,
            static_geometry: false,
            layer: 0,
            parent,
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            components,
        });
        scene.objects.len() - 1
    }

    fn has<T: 'static>(scene: &LoadedScene, i: usize) -> bool {
        scene.objects[i].components.iter().any(|c| c.as_any().is::<T>())
    }

    #[test]
    fn entering_play_builds_the_avatar_in_the_live_scene() {
        let dir = std::env::temp_dir().join(format!("zest-avatar-play-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let registry = ComponentRegistry::with_builtins();
        let mut s = LoadedScene::empty();
        let root = obj(&mut s, "Avatar", None, vec![Box::new(AvatarDescriptor::default()), Box::new(Animator::default())]);
        let hat = obj(
            &mut s,
            "Hat",
            Some(root),
            vec![Box::new(MenuToggle {
                menu_path: "Outfit/Hat".into(),
                actions: vec![ToggleAction::Active { target: "Hat".into(), on: true }],
                ..Default::default()
            })],
        );
        let body = obj(&mut s, "Body", Some(root), vec![]);
        let reports = build_for_play(&mut s, &dir, &ShaderLibrary::default(), &registry, 256);
        assert_eq!(reports.len(), 1);
        assert!(reports[0].errors.is_empty(), "{:?}", reports[0].errors);
        assert!(reports[0].layers >= 2 && reports[0].generated_params == 1 && reports[0].synced_bits == 1);
        // The toggled object starts off; the authoring component is gone; the finished ones are in.
        assert!(!s.objects[hat].enabled && s.objects[body].enabled);
        assert!(!has::<MenuToggle>(&s, hat));
        assert!(has::<AvatarParameters>(&s, root) && has::<AvatarMenuData>(&s, root));
        let anim = s.objects[root].components.iter().find_map(|c| c.as_any().downcast_ref::<Animator>()).unwrap();
        assert!(anim.controller.ends_with("/controller.animctrl") && dir.join(&anim.controller).is_file());
        // A second Play over the already-built scene changes nothing.
        let again = build_for_play(&mut s, &dir, &ShaderLibrary::default(), &registry, 256);
        assert!(again.is_empty());
    }

    #[test]
    fn scenes_without_avatars_cost_nothing() {
        let registry = ComponentRegistry::with_builtins();
        let mut s = LoadedScene::empty();
        obj(&mut s, "Cube", None, vec![]);
        assert!(build_for_play(&mut s, Path::new("."), &ShaderLibrary::default(), &registry, 256).is_empty());
    }

    /// Run the built avatar through the real play loop: the toggle parameter
    /// shows and hides the object through the generated controller.
    #[test]
    fn the_generated_controller_drives_the_toggle_in_play() {
        use crate::play::{PlaySession, TickInput, World};
        use zestengine_core::{InputState, NetView, ScreenInfo};

        let dir = std::env::temp_dir().join(format!("zest-avatar-play-loop-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let registry = ComponentRegistry::with_builtins();
        let mut s = LoadedScene::empty();
        let root = obj(&mut s, "Avatar", None, vec![Box::new(AvatarDescriptor::default()), Box::new(Animator::default())]);
        let hat = obj(
            &mut s,
            "Hat",
            Some(root),
            vec![Box::new(MenuToggle { menu_path: "Hat".into(), actions: vec![ToggleAction::Active { target: "Hat".into(), on: true }], ..Default::default() })],
        );
        let reports = build_for_play(&mut s, &dir, &ShaderLibrary::default(), &registry, 256);
        assert!(reports[0].errors.is_empty());
        assert!(!s.objects[hat].enabled);
        let mut play = PlaySession::new(&s, &dir, None);
        let mut shaders = ShaderLibrary::default();
        let (input, screen) = (InputState::default(), ScreenInfo::default());
        let net = NetView::default();
        let mut frame = |scene: &mut LoadedScene, play: &mut PlaySession, n: usize| {
            for _ in 0..n {
                play.tick(
                    &mut World { scene, renderer: None, registry: &registry, shaders: &mut shaders },
                    &TickInput { dt: 1.0 / 60.0, input: &input, net: &net, listener_fallback: glam::Mat4::IDENTITY, screen: &screen },
                );
            }
        };
        let set = |scene: &mut LoadedScene, v: f32| {
            let ap = scene.objects[root].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<AvatarParameters>()).unwrap();
            if !ap.ready {
                ap.rebuild_store();
            }
            let slot = (0..ap.store.len()).find(|&i| ap.store.name(i).starts_with("Tgl_Hat_")).unwrap();
            ap.store.set(slot, v);
        };
        frame(&mut s, &mut play, 5);
        assert!(!s.objects[hat].enabled, "off until the parameter is set");
        // Something else switching it on is undone by the defaults layer.
        s.objects[hat].enabled = true;
        frame(&mut s, &mut play, 5);
        assert!(!s.objects[hat].enabled, "the defaults layer holds the resting value");
        set(&mut s, 1.0);
        frame(&mut s, &mut play, 10);
        assert!(s.objects[hat].enabled, "the toggle layer turned the hat on");
        set(&mut s, 0.0);
        frame(&mut s, &mut play, 10);
        assert!(!s.objects[hat].enabled, "and the defaults layer put it back off");
    }
}
