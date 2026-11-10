//! Runtime mod lifecycle shared by desktop/VR games, editor Play and servers.
use std::path::{Path,PathBuf};
use anyhow::Result;
use zestengine_core::{ComponentRegistry,mods::{ModSettings,ModLock}};

pub fn settings(root:&Path)->ModSettings{zestengine_assets::load_project_file(root).map(|p|p.mods).unwrap_or_default()}
pub fn library(root:&Path,settings:&ModSettings)->PathBuf{
    if let Some(path)=std::env::var_os("ZESTENGINE_MOD_DIR"){return PathBuf::from(path);}
    if !zestengine_core::mods::valid_id(&settings.game_id){return root.join(".zestengine/mods");}
    let base=std::env::var_os("XDG_DATA_HOME").map(PathBuf::from)
        .or_else(||std::env::var_os("APPDATA").map(PathBuf::from))
        .or_else(||std::env::var_os("HOME").map(|p|PathBuf::from(p).join(".local/share")))
        .unwrap_or_else(||root.join(".zestengine"));
    base.join("zestengine").join(&settings.game_id).join("mods")
}
pub fn profile_name(settings:&ModSettings,library:&Path)->String{std::env::var("ZESTENGINE_MOD_PROFILE").ok().or_else(||std::fs::read_to_string(library.join("active.profile")).ok().map(|s|s.trim().to_owned())).unwrap_or_else(||settings.profile.clone())}
pub fn prepare(root:&Path,registry:&ComponentRegistry)->Result<Option<zestengine_assets::mods::ModMount>>{
    if root.join(zestengine_assets::mods::LOCK_FILE).exists(){lock(root)?;return Ok(None);}
    let settings=settings(root);let library=library(root,&settings);
    let profile=zestengine_assets::mods::load_profile(&library,&profile_name(&settings,&library))?;
    let known:std::collections::HashSet<_>=registry.names().collect();
    let mount=zestengine_assets::mods::mount(root,&library,&settings,&profile,&|name|known.contains(name))?;
    if let Some(m)=&mount{for warning in &m.warnings{tracing::warn!("mods: {warning}");}tracing::info!("Mounted {} mods for {}",m.lock.mods.len(),m.lock.game_id);}
    Ok(mount)
}
pub fn lock(root:&Path)->Result<ModLock>{zestengine_assets::mods::session_lock(root)}
