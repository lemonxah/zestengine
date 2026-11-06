    use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};
use zestengine_core::{ComponentRegistry, ContentKind};
use zestengine_assets::SceneFile;
use zestengine_engine::scene::LoadedScene;
use zestengine_engine::shaders::ShaderLibrary;
    use zestengine_assets::package;
use zestengine_assets::package::{BuildOptions, ComponentStatus, ContentLimits, PackagePlan};

    /// What the build window needs from the editor at click time.
    pub struct BuildSource<'a> {
        pub scene: &'a LoadedScene,
        pub shaders: &'a ShaderLibrary,
        pub root: &'a Path,
        pub selected: Option<usize>,
        pub scene_path: Option<&'a Path>,
        pub registry: &'a ComponentRegistry,
    }

    pub struct BuildSummary {
        pub out: PathBuf,
        pub plan: PackagePlan,
        pub written: bool,
    }

    pub struct BuildWindow {
        pub open: bool,
        kind: ContentKind,
        name: String,
        author: String,
        version: String,
        description: String,
        thumbnail: String,
        out: String,
        strip_plugins: bool,
        texture_cache: bool,
        running: Option<Receiver<Result<BuildSummary, String>>>,
        last: Option<Result<BuildSummary, String>>,
    }

    impl Default for BuildWindow {
        fn default() -> Self {
            Self {
                open: false,
                kind: ContentKind::World,
                name: String::new(),
                author: String::new(),
                version: "1.0".into(),
                description: String::new(),
                thumbnail: String::new(),
                out: String::new(),
                strip_plugins: false,
                texture_cache: true,
                running: None,
                last: None,
            }
        }
    }

    /// Classify component kinds for the target game: engine built-ins exist
    /// everywhere, the rest of the editor registry comes from plugins.
    pub fn classifier(registry: &ComponentRegistry) -> impl Fn(&str) -> ComponentStatus + Send + Sync + 'static {
        let builtin: HashSet<String> = ComponentRegistry::with_builtins().names().map(str::to_owned).collect();
        let all: HashSet<String> = registry.names().map(str::to_owned).collect();
        move |k: &str| {
            if builtin.contains(k) {
                ComponentStatus::Builtin
            } else if all.contains(k) {
                ComponentStatus::Plugin
            } else {
                ComponentStatus::Unknown
            }
        }
    }

    /// The entry scene + package path for a kind, from the live editor scene.
    pub fn source_scene(kind: ContentKind, src: &BuildSource) -> Result<(SceneFile, String), String> {
        match kind {
            ContentKind::World => {
                let file = src.scene.to_scene_file(src.root, src.shaders);
                let entry = src
                    .scene_path
                    .and_then(|p| p.strip_prefix(src.root).ok())
                    .map(|p| p.to_string_lossy().replace('\\', "/"))
                    .unwrap_or_else(|| "scenes/world.scene".into());
                Ok((file, entry))
            }
            ContentKind::Avatar | ContentKind::Prop => {
                let index = src.selected.ok_or("select the root object of the avatar / prop first")?;
                let prefab = src
                    .scene
                    .prefab_from_object(index, src.root, src.shaders)
                    .ok_or("selection is not a scene object")?;
                let mut entries = prefab.entries;
                if let Some(root) = entries.iter_mut().find(|e| e.parent.is_none()) {
                    root.translation = [0.0; 3];
                    root.rotation = [0.0, 0.0, 0.0, 1.0];
                }
                let name = src.scene.objects[index].name.clone();
                let stem: String = name
                    .chars()
                    .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
                    .collect();
                let file = SceneFile {
                    entries,
                    layers: src.scene.layers.clone(),
                    ..Default::default()
                };
                Ok((file, format!("content/{}.scene", if stem.is_empty() { "content" } else { &stem })))
            }
        }
    }

    impl BuildWindow {
        pub fn open_for(&mut self, kind: ContentKind) {
            self.open = true;
            self.kind = kind;
        }

        fn start(&mut self, src: &BuildSource, write: bool) {
            let (scene, entry) = match source_scene(self.kind, src) {
                Ok(v) => v,
                Err(e) => {
                    self.last = Some(Err(e));
                    return;
                }
            };
            let name = if self.name.trim().is_empty() {
                Path::new(&entry).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
            } else {
                self.name.clone()
            };
            let file_stem: String = name
                .chars()
                .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
                .collect();
            let out = if self.out.trim().is_empty() {
                src.root.join("build/content").join(format!("{file_stem}.{}", package::PACKAGE_EXTENSION))
            } else {
                PathBuf::from(self.out.trim())
            };
            let thumbnail = (!self.thumbnail.trim().is_empty())
                .then(|| std::fs::read(src.root.join(self.thumbnail.trim())).ok())
                .flatten();
            let classify = classifier(src.registry);
            let root = src.root.to_owned();
            let (kind, author, version, description) =
                (self.kind, self.author.clone(), self.version.clone(), self.description.clone());
            let (strip, texcache) = (self.strip_plugins, self.texture_cache);
            let (tx, rx) = channel();
            std::thread::spawn(move || {
                let opts = BuildOptions {
                    kind,
                    name,
                    author,
                    version,
                    description,
                    project_root: root.clone(),
                    entry,
                    thumbnail,
                    classify: &classify,
                    strip_plugin_components: strip,
                    include_texture_cache: texcache,
                    limits: ContentLimits::for_kind(kind),
                    scene,
                };
                let result = package::plan_package(&opts).map_err(|e| format!("{e:#}")).and_then(|mut plan| {
                    if kind == ContentKind::Avatar {
                        let scene_text = plan.files.get(&format!("files/{}", plan.manifest.entry)).cloned().unwrap_or_default();
                        if let Ok(file) = ron::from_str::<SceneFile>(&String::from_utf8_lossy(&scene_text)) {
                            plan.warnings.extend(zestengine_engine::content::avatar_warnings(&root, &file));
                        }
                    }
                    let written = write && plan.is_ok();
                    if written {
                        package::write_package(&plan, &out).map_err(|e| format!("{e:#}"))?;
                    }
                    Ok(BuildSummary { out, plan, written })
                });
                let _ = tx.send(result);
            });
            self.running = Some(rx);
            self.last = None;
        }

        pub fn show(&mut self, ctx: &egui::Context, src: &BuildSource) {
            if let Some(rx) = &self.running
                && let Ok(r) = rx.try_recv()
            {
                self.last = Some(r);
                self.running = None;
            }
            let mut open = self.open;
            egui::Window::new("Build Content Package")
                .open(&mut open)
                .default_width(460.0)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Kind");
                        for k in ContentKind::ALL {
                            ui.selectable_value(&mut self.kind, k, k.label());
                        }
                    });
                    let hint = match self.kind {
                        ContentKind::World => "Packages the current scene with its baked lighting.".to_owned(),
                        _ => match src.selected.and_then(|i| src.scene.objects.get(i)) {
                            Some(o) => format!("Packages {:?} and its children.", o.name),
                            None => "Select the root object in the hierarchy.".to_owned(),
                        },
                    };
                    ui.label(egui::RichText::new(hint).small().weak());
                    egui::Grid::new("zestengine-content-build").num_columns(2).show(ui, |ui| {
                        for (label, value, placeholder) in [
                            ("Name", &mut self.name, "My World"),
                            ("Author", &mut self.author, ""),
                            ("Version", &mut self.version, "1.0"),
                            ("Description", &mut self.description, ""),
                            ("Thumbnail", &mut self.thumbnail, "textures/thumb.png (optional)"),
                            ("Output", &mut self.out, "build/content/<name>.zestenginepak"),
                        ] {
                            ui.label(label);
                            ui.add(egui::TextEdit::singleline(value).hint_text(placeholder));
                            ui.end_row();
                        }
                    });
                    ui.checkbox(&mut self.texture_cache, "Include BC texture cache (faster loads, larger file)");
                    ui.checkbox(&mut self.strip_plugins, "Strip plugin components (keep only engine built-ins)");
                    crate::widgets::dialog_footer(ui, |ui| {
                        let idle = self.running.is_none();
                        ui.add_enabled_ui(idle, |ui| {
                            if crate::widgets::primary_button(ui, None, "Build").clicked() {
                                self.start(src, true);
                            }
                            if crate::widgets::secondary_button(ui, None, "Analyse").clicked() {
                                self.start(src, false);
                            }
                        });
                        if !idle {
                            ui.spinner();
                        }
                    });
                    ui.add_space(4.0);
                    match &self.last {
                        None => {}
                        Some(Err(e)) => {
                            ui.colored_label(egui::Color32::from_rgb(230, 100, 90), e);
                        }
                        Some(Ok(s)) => {
                            let m = &s.plan.manifest;
                            if s.written {
                                ui.label(format!("Wrote {}", s.out.display()));
                            }
                            ui.label(format!(
                                "{} files, {:.2} MiB, {} triangles, {:.1} MiB textures, {} objects",
                                m.files.len(),
                                m.stats.file_bytes as f64 / (1024.0 * 1024.0),
                                m.stats.triangles,
                                m.stats.texture_bytes as f64 / (1024.0 * 1024.0),
                                m.stats.objects
                            ));
                            ui.label(egui::RichText::new(format!("id {}", m.short_id())).small().weak());
                            ui.label(format!("Components: {}", if m.components.is_empty() { "none".into() } else { m.components.join(", ") }));
                            egui::ScrollArea::vertical().max_height(180.0).show(ui, |ui| {
                                for e in &s.plan.errors {
                                    ui.colored_label(egui::Color32::from_rgb(230, 100, 90), format!("error: {e}"));
                                }
                                for w in &s.plan.warnings {
                                    ui.colored_label(egui::Color32::from_rgb(230, 190, 90), format!("warning: {w}"));
                                }
                            });
                        }
                    }
                });
            self.open = open;
        }
    }
