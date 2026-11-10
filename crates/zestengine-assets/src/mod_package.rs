//! Mod authoring, installation, profiles and immutable per-session asset mounts.
//! Uses the existing content archive and validators; no second archive format.
use super::*;
use zestengine_core::mods::{ModManifest,ModSettings,ModProfile,ModLock,LockedMod};
use anyhow::ensure;
pub const MOD_MANIFEST:&str="mod.ron";
pub const LOCK_FILE:&str=".zest-mod-lock.ron";

pub(super) fn validate_module(bytes:&[u8],limits:&ContentLimits)->std::result::Result<(),String>{
    let sandbox=zestengine_core::wasm::SandboxLimits{max_module_bytes:limits.max_wasm_bytes.min(usize::MAX as u64)as usize,..zestengine_core::wasm::SandboxLimits::untrusted()};
    zestengine_core::wasm::inspect_module(bytes,&sandbox,zestengine_core::wasm::sandbox::ALL_CAPS).map(|_|())
}
fn checked_manifest(m:&ModManifest)->Result<()>{
    m.validate().map_err(anyhow::Error::msg)?;
    for target in m.overrides.keys().chain(m.additions.keys()){ensure!(!target.starts_with("_mods/")&&target!=MOD_MANIFEST,"Reserved mod path: {target}");}
    for (target,source)in &m.overrides{
        ensure!(ext_allowed(target)&&ext_of(target)==ext_of(source),"Override {target} must keep its asset type");
    }
    Ok(())
}
/// Build a mod with the normal scene/package authoring options. Additional
/// replacement assets and their dependencies are collected from the manifest.
pub fn plan_mod(opts:&BuildOptions,manifest:&ModManifest)->Result<PackagePlan>{
    checked_manifest(manifest)?;
    let mut plan=plan_package_with_mods(opts,true)?;
    let text=ron::ser::to_string_pretty(manifest,ron::ser::PrettyConfig::default())?;
    let (extra,warnings)=collect_dependencies(&opts.project_root,MOD_MANIFEST,&text);plan.warnings.extend(warnings);
    for rel in extra{
        if rel==MOD_MANIFEST{continue;}
        let path=opts.project_root.join(&rel);if !path.is_file(){continue;}
        ensure!(ext_allowed(&rel),"Unsupported mod file: {rel}");
        let bytes=std::fs::read(path)?;
        if ext_of(&rel)=="wasm"{validate_module(&bytes,&opts.limits).map_err(anyhow::Error::msg)?;}
        if ext_of(&rel)=="frag"{validate_shader_source(std::str::from_utf8(&bytes)?).map_err(anyhow::Error::msg)?;}
        plan.files.insert(format!("files/{rel}"),bytes);
    }
    for source in manifest.overrides.values().chain(manifest.additions.values()){
        ensure!(plan.files.contains_key(&format!("files/{source}")),"Missing mod asset: {source}");
    }
    ensure!(!plan.files.contains_key("files/mod.ron"),"mod.ron is reserved for the mod manifest");
    plan.files.insert("files/mod.ron".into(),text.into_bytes());
    plan.manifest.files=plan.files.iter().map(|(path,bytes)|FileRecord{path:path.clone(),size:bytes.len()as u64,hash:sha256_hex(bytes)}).collect();
    let mut components:BTreeSet<String>=plan.manifest.components.iter().cloned().collect();
    for (path,bytes)in &plan.files{if matches!(ext_of(path).as_str(),"scene"|"prefab"){
        let scene:SceneFile=ron::from_str(std::str::from_utf8(bytes)?)?;
        validate_entries(&scene.entries,&|_|true,opts.limits.max_objects)?;
        components.extend(scene.entries.iter().flat_map(|e|e.components.iter().map(|c|c.kind.clone())));
    }}
    plan.manifest.components=components.into_iter().collect();
    plan.manifest.stats.shaders=plan.files.keys().filter(|p|ext_of(p)=="frag").count()as u32;
    ensure!(plan.files.keys().filter(|p|ext_of(p)=="wasm").count()<=opts.limits.max_wasm_modules,"Too many WASM modules");
    plan.manifest.content_hash=content_hash(&plan.manifest.files);
    plan.manifest.stats.file_bytes=plan.manifest.files.iter().map(|f|f.size).sum();
    validate_manifest(&plan.manifest,&opts.limits)?;
    Ok(plan)
}
#[derive(Clone,Debug)]
pub struct InstalledMod{pub manifest:ModManifest,pub hash:String,pub path:PathBuf}
fn inspect(path:&Path)->Result<InstalledMod>{
    let mut reader=PackageReader::open(path,&ContentLimits::for_kind)?;
    let bytes=reader.read_asset(MOD_MANIFEST)?;ensure!(bytes.len()<=256*1024,"Mod manifest exceeds 256 KiB");
    let manifest:ModManifest=ron::from_str(std::str::from_utf8(&bytes)?)?;checked_manifest(&manifest)?;
    // Verify every payload now; archive metadata is not enough for a session identity.
    for record in reader.manifest.files.clone(){reader.read_file(&record.path)?;}
    Ok(InstalledMod{manifest,hash:reader.manifest.content_hash.clone(),path:path.to_owned()})
}
/// One install per ID; replacing a version is atomic and requires a new session.
pub fn install(source:&Path,library:&Path)->Result<InstalledMod>{
    std::fs::create_dir_all(library)?;
    let temp=library.join(format!("install-{}.tmp",zestengine_core::ObjectId::new()));
    let result=(||{std::fs::copy(source,&temp)?;let mut entry=inspect(&temp)?;let dest=library.join(format!("{}.zestenginepak",entry.manifest.id));std::fs::rename(&temp,&dest)?;entry.path=dest;Ok(entry)})();
    if result.is_err(){let _=std::fs::remove_file(temp);}result
}
pub fn installed(library:&Path)->Result<BTreeMap<String,InstalledMod>>{
    let mut out=BTreeMap::new();if !library.exists(){return Ok(out);}
    let mut files=std::fs::read_dir(library)?.collect::<std::io::Result<Vec<_>>>()?;files.sort_by_key(|e|e.file_name());
    for entry in files{if entry.path().extension().is_some_and(|e|e==PACKAGE_EXTENSION){let m=inspect(&entry.path()).with_context(||format!("checking {}",entry.path().display()))?;ensure!(!out.contains_key(&m.manifest.id),"Duplicate installed mod: {}",m.manifest.id);out.insert(m.manifest.id.clone(),m);}}
    Ok(out)
}
pub fn profile_path(library:&Path,name:&str)->Result<PathBuf>{ensure!(zestengine_core::mods::valid_id(name),"Invalid mod profile name");Ok(library.join("profiles").join(format!("{name}.ron")))}
pub fn load_profile(library:&Path,name:&str)->Result<ModProfile>{let path=profile_path(library,name)?;if !path.exists(){return Ok(ModProfile::default());}ensure!(std::fs::metadata(&path)?.len()<64*1024,"Mod profile too large");Ok(ron::from_str(&std::fs::read_to_string(path)?)?)}
pub fn save_profile(library:&Path,name:&str,profile:&ModProfile)->Result<()>{
    let path=profile_path(library,name)?;std::fs::create_dir_all(path.parent().unwrap())?;let tmp=path.with_extension(format!("{}.tmp",zestengine_core::ObjectId::new()));
    std::fs::write(&tmp,ron::ser::to_string_pretty(profile,ron::ser::PrettyConfig::default())?)?;if let Err(e)=std::fs::rename(&tmp,&path){let _=std::fs::remove_file(tmp);return Err(e.into());}Ok(())
}
#[derive(Clone,Debug,Serialize,Deserialize)]
struct SessionRecord{origin:PathBuf,lock:ModLock}
pub fn session_lock(root:&Path)->Result<ModLock>{
    if !root.join(LOCK_FILE).exists(){return Ok(ModLock::default());}
    let record:SessionRecord=ron::from_str(&std::fs::read_to_string(root.join(LOCK_FILE))?)?;Ok(record.lock)
}
pub fn origin_root(root:&Path)->PathBuf{std::fs::read_to_string(root.join(LOCK_FILE)).ok().and_then(|s|ron::from_str::<SessionRecord>(&s).ok()).map_or_else(||root.to_owned(),|r|r.origin)}
/// Owns a read snapshot for a running game. Changes to installations/profiles
/// apply to the next session; they cannot change an admitted multiplayer game.
pub struct ModMount{pub root:PathBuf,pub lock:ModLock,pub warnings:Vec<String>,roots:Vec<PathBuf>}
impl Drop for ModMount{fn drop(&mut self){for root in &self.roots{zestengine_core::wasm::unregister_mod_root(root);}let _=std::fs::remove_dir_all(&self.root);}}
fn copy_assets(source:&Path,dest:&Path)->Result<()>{
    std::fs::create_dir_all(dest)?;
    for entry in std::fs::read_dir(source)?{
        let entry=entry?;let name=entry.file_name();let name=name.to_string_lossy();
        if name.starts_with('.')||matches!(name.as_ref(),"target"|"build"|"plugins"|"node_modules"|"_mods"){continue;}
        let kind=entry.file_type()?;
        ensure!(!kind.is_symlink(),"Mod mounts require assets inside the project, without symlinks: {}",entry.path().display());
        if kind.is_dir(){copy_assets(&entry.path(),&dest.join(name.as_ref()))?;}
        else if kind.is_file(){std::fs::copy(entry.path(),dest.join(name.as_ref()))?;}
    }
    Ok(())
}
fn validate_entries(entries:&[crate::SceneEntry],known:&dyn Fn(&str)->bool,max_objects:usize)->Result<()>{
    ensure!(entries.len()<=max_objects,"Mod scene exceeds object limit");
    let mut ids=BTreeSet::new();
    for (i,e)in entries.iter().enumerate(){
        if !e.id.is_empty(){ensure!(ids.insert(&e.id),"Duplicate object ID in mod scene");}
        let mut parent=e.parent;let mut visited=BTreeSet::from([i]);
        while let Some(p)=parent{ensure!(p<entries.len()&&visited.insert(p),"Invalid/cyclic mod scene parent");parent=entries[p].parent;}
        for c in &e.components{ensure!(c.kind!="Script","Mods must use Wasm Script, not the legacy Script prototype");ensure!(known(&c.kind),"Mod needs unavailable component {}",c.kind);}
    }Ok(())
}
pub(super) fn prepare_scene(text:&str,known:&dyn Fn(&str)->bool,max_objects:usize)->Result<String>{
    let mut scene:SceneFile=ron::from_str(text)?;validate_entries(&scene.entries,known,max_objects)?;
    for e in &mut scene.entries{
        let mut instance=0;
        for c in &e.components{if c.kind=="Wasm Script"{let key=if instance==0{c.kind.clone()}else{format!("{}#{instance}",c.kind)};e.component_roles.insert(key,zestengine_core::NetRole::Server);instance+=1;}}
    }
    Ok(ron::ser::to_string_pretty(&scene,ron::ser::PrettyConfig::default())?)
}
fn append_scene(target:&Path,source:&Path,id:&str,target_rel:&str)->Result<()>{
    let mut base=crate::load_scene_file(target)?;let mut extra=crate::load_scene_file(source)?;
    let count=extra.entries.len();let offset=base.entries.len();let mut ids=BTreeMap::new();
    for (i,e)in extra.entries.iter().enumerate(){
        ensure!(e.parent.is_none_or(|p|p<count&&p!=i),"Invalid mod scene parent");
        let key=format!("mod-object-v1:{id}:{target_rel}:{i}:{}",e.id);let hash=sha256::digest(key.as_bytes());let new=zestengine_core::ObjectId::from_raw(u128::from_le_bytes(hash[..16].try_into().unwrap())).to_string();
        if !e.id.is_empty(){ensure!(ids.insert(e.id.clone(),new).is_none(),"Duplicate object ID in mod scene");}
    }
    for (i,e)in extra.entries.iter_mut().enumerate(){
        let key=format!("mod-object-v1:{id}:{target_rel}:{i}:{}",e.id);let hash=sha256::digest(key.as_bytes());
        e.id=zestengine_core::ObjectId::from_raw(u128::from_le_bytes(hash[..16].try_into().unwrap())).to_string();
        if let Some(parent)=&mut e.parent{*parent+=offset;}
        for c in &mut e.components{c.data=rewrite_ron_strings(&c.data,&mut |value|ids.get(value).cloned());}
        if let Some(link)=&mut e.prefab{for o in &mut link.overrides{o.value=rewrite_ron_strings(&o.value,&mut |value|ids.get(value).cloned().or_else(||value.strip_prefix("ext:").and_then(|id|ids.get(id)).map(|new|format!("ext:{new}"))));}}
        if let Some(parent)=&mut e.prefab_parent{if let Some(new)=ids.get(parent){*parent=new.clone();}}
        // Gameplay modules run authoritatively. Client views use ordinary replicated components.
        for c in &e.components{ensure!(c.kind!="Script","Mods must use Wasm Script, not the legacy Script prototype");}

    }
    base.entries.extend(extra.entries);
    std::fs::write(target,ron::ser::to_string_pretty(&base,ron::ser::PrettyConfig::default())?)?;Ok(())
}
/// Prepare before loading the boot scene or opening a network connection.
/// Every asset consumer uses this root, so replacements apply consistently to
/// rendering, physics, audio, prefabs, scripts and scene transitions.
pub fn mount(root:&Path,library:&Path,settings:&ModSettings,profile:&ModProfile,known:&dyn Fn(&str)->bool)->Result<Option<ModMount>>{
    if profile.enabled.is_empty(){return Ok(None);}
    ensure!(settings.enabled,"This game disables mods");
    let entries=installed(library)?;let manifests=entries.iter().map(|(id,m)|(id.clone(),m.manifest.clone())).collect();
    let ordered=zestengine_core::mods::resolve(profile,&manifests,settings).map_err(anyhow::Error::msg)?;
    let lock=ModLock{game_id:settings.game_id.clone(),game_api:settings.game_api,mods:ordered.iter().map(|m|LockedMod{id:m.id.clone(),version:m.version.clone(),content_hash:entries[&m.id].hash.clone()}).collect()};
    let destination=std::env::temp_dir().join(format!("zest-mod-session-{}",zestengine_core::ObjectId::new()));
    let mut mounted=ModMount{root:destination,lock,warnings:Vec::new(),roots:Vec::new()};
    copy_assets(root,&mounted.root)?;
    rebase_base_files(&mounted.root,&std::fs::canonicalize(root)?,&mounted.root)?;
    crate::import_hooks::mark_untrusted_root(&mounted.root);
    let mut overridden=BTreeMap::<String,String>::new();
    for manifest in ordered{
        let installed=&entries[&manifest.id];let mut reader=PackageReader::open(&installed.path,&ContentLimits::for_kind)?;
        ensure!(reader.manifest.content_hash==installed.hash,"Mod changed while mounting: {}",manifest.id);
        for component in &reader.manifest.components{ensure!(known(component),"{} needs unavailable component {component}",manifest.id);}
        let cache=ContentCache::new(mounted.root.join("_mods").join(&manifest.id));
        let ex=cache.extract_with_mods(&mut reader,&ExtractOptions{known_component:known,components_fingerprint:"mod-v1".into(),server_side:false},true)?;
        let old_files=ex.dir.join("files");let files=mounted.root.join("_mods").join(&manifest.id).join("files");
        std::fs::rename(&old_files,&files)?;
        rebase_files(&files,&old_files,&files,&mounted.root)?;mounted.warnings.extend(ex.warnings);
        zestengine_core::wasm::register_mod_root(&files,manifest.actions.clone());mounted.roots.push(files.clone());
        for (target,source)in &manifest.overrides{
            ensure!(files.join(source).is_file(),"Missing replacement: {source}");
            if let Some(previous)=overridden.insert(target.clone(),manifest.id.clone()){mounted.warnings.push(format!("{target}: {} overrides {previous}",manifest.id));}
            let dest=mounted.root.join(target);std::fs::create_dir_all(dest.parent().unwrap())?;copy_override(&files.join(source),&dest)?;
            // A replacement WASM retains its sandbox after moving to a base-game path.
            if ext_of(target)=="wasm"{zestengine_core::wasm::register_mod_root(&dest,manifest.actions.clone());mounted.roots.push(dest);}
        }
        for (target,source)in &manifest.additions{append_scene(&mounted.root.join(target),&files.join(source),&manifest.id,target)?;}
    }
    std::fs::write(mounted.root.join(LOCK_FILE),ron::to_string(&SessionRecord{origin:std::fs::canonicalize(root)?,lock:mounted.lock.clone()})?)?;
    Ok(Some(mounted))
}

// Preserve base-game references when running a snapshot outside the authored project.
// WASM opens files directly; convert its relative/GUID references to the snapshot.
fn rebase_base_files(dir:&Path,original:&Path,root:&Path)->Result<()>{
    for e in std::fs::read_dir(dir)?{let e=e?;if e.file_type()?.is_dir(){rebase_base_files(&e.path(),original,root)?;continue;}
        let path=e.path();if !is_ron_text(&path.to_string_lossy()){continue;}
        let text=std::fs::read_to_string(&path)?;
        let rewritten=rewrite_ron_strings(&text,&mut |value|{
            let resolved=zestengine_core::asset_ref::resolve(root,value)?;
            let p=Path::new(&resolved);
            let relative=p.strip_prefix(original).ok().or_else(||(!p.is_absolute()).then_some(p))?;
            if ext_of(&resolved)=="wasm"{Some(root.join(relative).to_string_lossy().into_owned())}
            else if p.is_absolute(){Some(relative.to_string_lossy().replace('\\',"/"))}else{None}
        });
        if rewritten!=text{std::fs::write(path,rewritten)?;}
    }Ok(())
}
fn rebase_files(dir:&Path,old:&Path,new:&Path,root:&Path)->Result<()>{
    for e in std::fs::read_dir(dir)?{let e=e?;if e.file_type()?.is_dir(){rebase_files(&e.path(),old,new,root)?;continue;}
        let path=e.path();let rel=path.to_string_lossy();if !is_ron_text(&rel){continue;}
        let text=std::fs::read_to_string(&path)?;
        let text=rewrite_ron_strings(&text,&mut |s|{
            let suffix=Path::new(s).strip_prefix(old).ok()?;let target=new.join(suffix);
            // The WASM component opens an absolute file. Other references,
            // including networked prefab paths, must be identical on every peer.
            Some(if ext_of(s)=="wasm"{target.to_string_lossy().into_owned()}else{target.strip_prefix(root).ok()?.to_string_lossy().replace('\\',"/")})
        });std::fs::write(path,text)?;
    }Ok(())
}
fn copy_override(source:&Path,dest:&Path)->Result<()>{
    if source.extension().is_some_and(|e|e=="gltf"){
        let mut json:serde_json::Value=serde_json::from_slice(&std::fs::read(source)?)?;
        for category in ["buffers","images"]{if let Some(entries)=json[category].as_array_mut(){for entry in entries{
            if let Some(uri)=entry["uri"].as_str().filter(|s|!s.starts_with("data:")){
                let target=source.parent().unwrap().join(uri);
                let from=dest.parent().unwrap().components().collect::<Vec<_>>();let to=target.components().collect::<Vec<_>>();let common=from.iter().zip(&to).take_while(|(a,b)|a==b).count();
                let mut relative=PathBuf::new();for _ in common..from.len(){relative.push("..");}for part in &to[common..]{relative.push(part.as_os_str());}
                entry["uri"]=serde_json::Value::String(relative.to_string_lossy().replace('\\',"/"));
            }
        }}}
        std::fs::write(dest,serde_json::to_vec(&json)?)?;
    }else{std::fs::copy(source,dest)?;}Ok(())
}

#[cfg(test)]mod tests{
    use super::*;
    struct Temp(PathBuf);impl Temp{fn new()->Self{let p=std::env::temp_dir().join(format!("zest-mod-test-{}",zestengine_core::ObjectId::new()));std::fs::create_dir_all(&p).unwrap();Self(p)}}impl Drop for Temp{fn drop(&mut self){let _=std::fs::remove_dir_all(&self.0);}}
    fn scene(name:&str)->SceneFile{ron::from_str(&format!("(entries:[(id:\"{}\",name:\"{name}\",source:Empty,translation:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),scale:(1.0,1.0,1.0))])",zestengine_core::ObjectId::new())).unwrap()}
    fn write_scene(path:&Path,scene:&SceneFile){std::fs::create_dir_all(path.parent().unwrap()).unwrap();std::fs::write(path,ron::to_string(scene).unwrap()).unwrap();}
    fn opts(root:&Path,scene:SceneFile)->BuildOptions<'static>{BuildOptions{kind:ContentKind::World,name:"Test mod".into(),author:"test".into(),version:"1.0.0".into(),description:String::new(),project_root:root.to_owned(),scene,entry:"addon.scene".into(),thumbnail:None,classify:&|_|ComponentStatus::Builtin,strip_plugin_components:false,include_texture_cache:false,limits:ContentLimits::for_kind(ContentKind::World)}}
    #[test]fn mod_package_install_mount_and_profile_preserve_identity_and_base_files(){
        let t=Temp::new();let game=t.0.join("game");let author=t.0.join("author");let library=t.0.join("library");std::fs::create_dir_all(&author).unwrap();std::fs::create_dir_all(game.join("settings")).unwrap();
        write_scene(&game.join("main.scene"),&scene("Base"));std::fs::write(game.join("settings/items.ron"),"(damage:1)").unwrap();std::fs::write(author.join("items.ron"),"(damage:10)").unwrap();
        let mut addon=scene("Mod object");addon.entries[0].components.push(crate::ComponentData::new("Network Object",ron::to_string(&zestengine_core::NetworkObject::default()).unwrap()));write_scene(&author.join("addon.scene"),&addon);
        let mut manifest=ModManifest::new("test",1);manifest.name="Test mod".into();manifest.overrides.insert("settings/items.ron".into(),"items.ron".into());manifest.additions.insert("main.scene".into(),"addon.scene".into());
        let plan=plan_mod(&opts(&author,addon),&manifest).unwrap();let archive=t.0.join("mod.zestenginepak");write_package(&plan,&archive).unwrap();let entry=install(&archive,&library).unwrap();assert_eq!(entry.manifest.id,manifest.id);
        let profile=ModProfile{enabled:vec![manifest.id.clone()]};save_profile(&library,"default",&profile).unwrap();assert_eq!(load_profile(&library,"default").unwrap(),profile);
        let settings=ModSettings{game_id:"test".into(),override_paths:vec!["main.scene".into(),"settings/".into()],..Default::default()};
        let a=mount(&game,&library,&settings,&profile,&|_|true).unwrap().unwrap();let b=mount(&game,&library,&settings,&profile,&|_|true).unwrap().unwrap();
        assert_eq!(a.lock,b.lock);assert_ne!(a.root,b.root);assert_eq!(session_lock(&a.root).unwrap(),a.lock);assert_eq!(std::fs::read_to_string(a.root.join("settings/items.ron")).unwrap(),"(damage:10)");assert_eq!(std::fs::read_to_string(game.join("settings/items.ron")).unwrap(),"(damage:1)");
        let sa=crate::load_scene_file(a.root.join("main.scene")).unwrap();let sb=crate::load_scene_file(b.root.join("main.scene")).unwrap();assert_eq!(sa.entries.len(),2);assert_eq!(sa.entries[1].id,sb.entries[1].id);assert_ne!(sa.entries[0].id,sa.entries[1].id);
        let root=a.root.clone();drop(a);assert!(!root.exists());assert!(game.join("main.scene").exists());
        let installed=library.join(format!("{}.zestenginepak",manifest.id));std::fs::write(&installed,b"broken").unwrap();assert!(mount(&game,&library,&settings,&profile,&|_|true).is_err());
        // A running session remains intact after an installation changes.
        assert_eq!(crate::load_scene_file(b.root.join("main.scene")).unwrap().entries.len(),2);
    }
    #[test]fn mod_mount_preserves_base_assets_and_rebases_base_script_references(){
        let t=Temp::new();let original=t.0.join("original");let snapshot=t.0.join("snapshot");std::fs::create_dir_all(original.join("logic")).unwrap();
        std::fs::write(original.join("font.ttf"),b"font").unwrap();std::fs::write(original.join("logic/base.wasm"),b"module").unwrap();
        let mut scene=scene("Script");scene.entries[0].components.push(crate::ComponentData::new("Wasm Script","(module:\"logic/base.wasm\")"));
        write_scene(&original.join("main.scene"),&scene);copy_assets(&original,&snapshot).unwrap();rebase_base_files(&snapshot,&original,&snapshot).unwrap();
        assert_eq!(std::fs::read(snapshot.join("font.ttf")).unwrap(),b"font");
        let loaded=crate::load_scene_file(snapshot.join("main.scene")).unwrap();assert!(loaded.entries[0].components[0].data.contains(snapshot.join("logic/base.wasm").to_str().unwrap()));
        assert!(crate::load_scene_file(original.join("main.scene")).unwrap().entries[0].components[0].data.contains("logic/base.wasm"));
    }
    #[test]fn mod_scene_validation_rejects_cycles_unknown_components_and_prototype_scripts(){
        let mut scene=scene("Script");scene.entries[0].components.push(crate::ComponentData::new("Wasm Script","(module:\"logic.wasm\")"));
        let duplicate=scene.entries[0].components[0].clone();scene.entries[0].components.push(duplicate);
        let text=ron::to_string(&scene).unwrap();let prepared=prepare_scene(&text,&|name|name=="Wasm Script",10).unwrap();
        let prepared:SceneFile=ron::from_str(&prepared).unwrap();assert_eq!(prepared.entries[0].component_roles["Wasm Script#1"],zestengine_core::NetRole::Server);
        assert!(prepare_scene(&text,&|_|false,10).is_err());
        scene.entries[0].parent=Some(0);assert!(prepare_scene(&ron::to_string(&scene).unwrap(),&|_|true,10).is_err());
        scene.entries[0].parent=None;scene.entries[0].components[0].kind="Script".into();assert!(prepare_scene(&ron::to_string(&scene).unwrap(),&|_|true,10).is_err());
    }
    #[test]fn mod_payloads_have_portable_paths_and_game_permissions_cannot_be_self_granted(){
        let t=Temp::new();let root=t.0.join("author");std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("thing.prefab"),"(entries:[])").unwrap();
        let mut scene=scene("Addon");scene.entries[0].components.push(crate::ComponentData::new("Spawner","(prefab:\"thing.prefab\")"));write_scene(&root.join("addon.scene"),&scene);
        let mut m=ModManifest::new("test",1);m.additions.insert("main.scene".into(),"addon.scene".into());m.actions.push("Health.Damage".into());
        let plan=plan_mod(&opts(&root,scene),&m).unwrap();let archive=t.0.join("mod.zestenginepak");write_package(&plan,&archive).unwrap();let library=t.0.join("installed");install(&archive,&library).unwrap();
        let game=t.0.join("game");write_scene(&game.join("main.scene"),&self::scene("Base"));let profile=ModProfile{enabled:vec![m.id.clone()]};let mut settings=ModSettings{game_id:"test".into(),override_paths:vec!["main.scene".into()],..Default::default()};
        assert!(mount(&game,&library,&settings,&profile,&|_|true).is_err());settings.actions.push("Health.Damage".into());
        let a=mount(&game,&library,&settings,&profile,&|_|true).unwrap().unwrap();let b=mount(&game,&library,&settings,&profile,&|_|true).unwrap().unwrap();
        let sa=crate::load_scene_file(a.root.join("main.scene")).unwrap();let sb=crate::load_scene_file(b.root.join("main.scene")).unwrap();assert_eq!(sa.entries[1].components,sb.entries[1].components);assert!(sa.entries[1].components[0].data.contains(&format!("_mods/{}/files/thing.prefab",m.id)));assert!(!sa.entries[1].components[0].data.contains("zest-mod-session"));
    }
}
