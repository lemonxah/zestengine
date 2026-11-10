//! Shared player/editor mod manager. Profile edits take effect next session.
use std::path::{Path,PathBuf};
use zestengine_core::mods::{ModSettings,ModProfile};
#[derive(Clone,Default)]
struct State {library:PathBuf,entries:Vec<zestengine_assets::mods::InstalledMod>,profile:ModProfile,name:String,source:String,status:String,build_manifest:String,build_scene:String}
fn refresh(s:&mut State,library:&Path,settings:&ModSettings){
    s.library=library.to_owned();s.name=super::modding::profile_name(settings,library);
    match zestengine_assets::mods::installed(library){Ok(v)=>s.entries=v.into_values().collect(),Err(e)=>{s.entries.clear();s.status=format!("{e:#}");}}
    match zestengine_assets::mods::load_profile(library,&s.name){Ok(p)=>s.profile=p,Err(e)=>s.status=format!("{e:#}")}
}
pub fn manager(ui:&mut egui::Ui,root:&Path,settings:&ModSettings,running:bool){
    let library=super::modding::library(&zestengine_assets::mods::origin_root(root),settings);
    let id=ui.make_persistent_id("mod-manager-state");let mut s=ui.data_mut(|d|d.get_temp::<State>(id)).unwrap_or_default();
    if s.library!=library{refresh(&mut s,&library,settings);}
    ui.label("Mods are shared by every player in a session. Versions, contents and load order must match.");
    if running{ui.label("Changes apply after restarting the game or stopping and restarting Play.");}
    ui.small(format!("Installed mods: {}",library.display()));
    ui.horizontal(|ui|{ui.label("Profile");ui.text_edit_singleline(&mut s.name);if ui.button("Load").clicked(){match zestengine_assets::mods::load_profile(&library,&s.name){Ok(p)=>s.profile=p,Err(e)=>s.status=format!("{e:#}")}}if ui.button("Refresh").clicked(){refresh(&mut s,&library,settings);}});
    let mut move_up=None;let mut move_down=None;
    let mut order=s.entries.iter().map(|m|m.manifest.id.clone()).collect::<Vec<_>>();order.sort_by_key(|id|(s.profile.enabled.iter().position(|v|v==id).unwrap_or(usize::MAX),id.clone()));
    for id in order{
        let entry=s.entries.iter().find(|m|m.manifest.id==id).unwrap();
        ui.horizontal(|ui|{
            let mut enabled=s.profile.enabled.contains(&id);if ui.checkbox(&mut enabled,format!("{}  {}",entry.manifest.name,entry.manifest.version)).changed(){if enabled{s.profile.enabled.push(id.clone());}else{s.profile.enabled.retain(|v|v!=&id);}}
            ui.small(&id);
            if enabled{if ui.small_button("↑").clicked(){move_up=Some(id.clone());}if ui.small_button("↓").clicked(){move_down=Some(id.clone());}}
        });
        if !entry.manifest.dependencies.is_empty(){ui.small(format!("Requires: {}",entry.manifest.dependencies.iter().map(|d|d.id.as_str()).collect::<Vec<_>>().join(", ")));}
    }
    for (id,up)in [(move_up,true),(move_down,false)]{if let Some(id)=id{if let Some(i)=s.profile.enabled.iter().position(|v|v==&id){let j=if up{i.saturating_sub(1)}else{(i+1).min(s.profile.enabled.len()-1)};s.profile.enabled.swap(i,j);}}}
    if ui.button("Validate and save profile").clicked(){
        let manifests=s.entries.iter().map(|m|(m.manifest.id.clone(),m.manifest.clone())).collect();
        let result=(||->anyhow::Result<()>{
            let ordered=zestengine_core::mods::resolve(&s.profile,&manifests,settings).map_err(anyhow::Error::msg)?;
            let mut overrides=std::collections::BTreeMap::new();let mut notices=Vec::new();
            for m in ordered{for path in m.overrides.keys(){if let Some(previous)=overrides.insert(path.clone(),m.id.clone()){notices.push(format!("{} overrides {previous}: {path}",m.id));}}}
            zestengine_assets::mods::save_profile(&library,&s.name,&s.profile)?;
            std::fs::write(library.join("active.profile"),&s.name)?;
            s.status=if notices.is_empty(){"Profile saved. It will be used next session.".into()}else{format!("Profile saved. {}",notices.join("; "))};Ok(())
        })();if let Err(e)=result{s.status=format!("{e:#}");}
    }
    ui.separator();ui.label("Install a mod package");ui.add(egui::TextEdit::singleline(&mut s.source).hint_text("Path to a .zestenginepak file"));
    if ui.button("Install / update").clicked(){match zestengine_assets::mods::install(Path::new(&s.source),&library){Ok(entry)=>{s.status=format!("Installed {} {}",entry.manifest.name,entry.manifest.version);refresh(&mut s,&library,settings);},Err(e)=>s.status=format!("{e:#}")}}
    if !s.status.is_empty(){ui.label(&s.status);}ui.data_mut(|d|d.insert_temp(id,s));
}
pub fn project_settings(ui:&mut egui::Ui,root:&Path,settings:&mut ModSettings,running:bool)->bool{
    let before=settings.clone();
    ui.add_enabled_ui(!running,|ui|{
        ui.checkbox(&mut settings.enabled,"Allow mods");
        ui.horizontal(|ui|{ui.label("Game ID");ui.text_edit_singleline(&mut settings.game_id);});
        ui.horizontal(|ui|{ui.label("Mod API version");ui.add(egui::DragValue::new(&mut settings.game_api).range(1..=u32::MAX));});
        ui.label("Exposed asset paths / directory prefixes (one per line)");let mut paths=settings.override_paths.join("\n");if ui.text_edit_multiline(&mut paths).changed(){settings.override_paths=paths.lines().map(str::trim).filter(|s|!s.is_empty()).map(str::to_owned).collect();}
        ui.label("Exposed script actions, Component.Action (one per line)");let mut actions=settings.actions.join("\n");if ui.text_edit_multiline(&mut actions).changed(){settings.actions=actions.lines().map(str::trim).filter(|s|!s.is_empty()).map(str::to_owned).collect();}
    });
    manager(ui,root,settings,running);
    ui.collapsing("Build a mod package",|ui|{
        let id=ui.make_persistent_id("mod-build");let mut s=ui.data_mut(|d|d.get_temp::<State>(id)).unwrap_or_default();
        ui.label("Manifest and entry scene are relative to this project.");ui.add(egui::TextEdit::singleline(&mut s.build_manifest).hint_text("my-mod.ron"));ui.add(egui::TextEdit::singleline(&mut s.build_scene).hint_text("scenes/mod.scene"));
        if ui.add_enabled(!running,egui::Button::new("Create mod manifest (new UUID)")).clicked(){
            let result=(||->anyhow::Result<String>{
                anyhow::ensure!(zestengine_core::mods::valid_id(&settings.game_id),"Set the target Game ID first");
                anyhow::ensure!(zestengine_core::mods::relative_asset(&s.build_manifest),"Enter a project-relative manifest filename");
                let m=zestengine_core::mods::ModManifest::new(&settings.game_id,settings.game_api);let path=root.join(&s.build_manifest);
                if let Some(parent)=path.parent(){std::fs::create_dir_all(parent)?;}
                use std::io::Write;let mut file=std::fs::OpenOptions::new().write(true).create_new(true).open(path)?;file.write_all(ron::ser::to_string_pretty(&m,ron::ser::PrettyConfig::default())?.as_bytes())?;
                Ok(format!("Created {}. Keep this UUID when releasing updates.",m.id))
            })();s.status=match result{Ok(message)=>message,Err(e)=>format!("{e:#}")};
        }
        if ui.add_enabled(!running,egui::Button::new("Build .zestenginepak")).clicked(){
            let result=(||->anyhow::Result<PathBuf>{
                anyhow::ensure!(zestengine_core::mods::relative_asset(&s.build_manifest)&&zestengine_core::mods::relative_asset(&s.build_scene),"Use project-relative paths");
                let manifest:zestengine_core::mods::ModManifest=ron::from_str(&std::fs::read_to_string(root.join(&s.build_manifest))?)?;
                let scene=zestengine_assets::load_scene_file(root.join(&s.build_scene))?;
                let opts=zestengine_assets::package::BuildOptions{kind:zestengine_core::ContentKind::World,name:manifest.name.clone(),author:manifest.author.clone(),version:manifest.version.clone(),description:String::new(),project_root:root.to_owned(),scene,entry:s.build_scene.clone(),thumbnail:None,classify:&|_|zestengine_assets::package::ComponentStatus::Builtin,strip_plugin_components:false,include_texture_cache:false,limits:zestengine_assets::package::ContentLimits::for_kind(zestengine_core::ContentKind::World)};
                let plan=zestengine_assets::mods::plan_mod(&opts,&manifest)?;let out=root.join("build/content").join(format!("{}.zestenginepak",manifest.id));std::fs::create_dir_all(out.parent().unwrap())?;zestengine_assets::package::write_package(&plan,&out)?;Ok(out)
            })();s.status=match result{Ok(path)=>format!("Built {}",path.display()),Err(e)=>format!("{e:#}")};
        }
        ui.label(&s.status);ui.data_mut(|d|d.insert_temp(id,s));
    });
    *settings!=before
}

/// The game HUD uses its own input model; standard egui fields need raw keys.
pub(crate) fn key_event(event:&winit::event::KeyEvent,modifiers:egui::Modifiers,out:&mut Vec<egui::Event>){
    use winit::keyboard::{Key,NamedKey};
    let key=match &event.logical_key{
        Key::Named(n)=>match n{NamedKey::ArrowLeft=>Some(egui::Key::ArrowLeft),NamedKey::ArrowRight=>Some(egui::Key::ArrowRight),NamedKey::ArrowUp=>Some(egui::Key::ArrowUp),NamedKey::ArrowDown=>Some(egui::Key::ArrowDown),NamedKey::Home=>Some(egui::Key::Home),NamedKey::End=>Some(egui::Key::End),NamedKey::Backspace=>Some(egui::Key::Backspace),NamedKey::Delete=>Some(egui::Key::Delete),NamedKey::Enter=>Some(egui::Key::Enter),NamedKey::Escape=>Some(egui::Key::Escape),NamedKey::Tab=>Some(egui::Key::Tab),_=>None},
        Key::Character(s) if s.eq_ignore_ascii_case("a")=>Some(egui::Key::A),_=>None,
    };
    if let Some(key)=key{out.push(egui::Event::Key{key,physical_key:None,pressed:event.state.is_pressed(),repeat:event.repeat,modifiers});}
    if event.state.is_pressed()&&!modifiers.command&&!modifiers.ctrl{
        if let Some(text)=event.text.as_ref().filter(|s|!s.chars().any(char::is_control)){out.push(egui::Event::Text(text.to_string()));}
    }
}
