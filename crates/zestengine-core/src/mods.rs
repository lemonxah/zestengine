//! Shared mod contracts. Filesystem/package handling belongs to assets; loading
//! and multiplayer admission belong to the engine. Profiles are frozen for a session.
use std::collections::{BTreeMap,BTreeSet};
use serde::{Serialize,Deserialize};

#[derive(Clone,Debug,PartialEq,Eq,Serialize,Deserialize)]
#[serde(default)]
pub struct ModSettings {
    pub enabled:bool,
    /// Stable identifier chosen by the game, independent of its display name.
    pub game_id:String,
    pub game_api:u32,
    pub profile:String,
    /// Exact asset paths or directory prefixes ending in `/` exposed by the game.
    pub override_paths:Vec<String>,
    /// Component.Action names scripts may invoke, explicitly exposed by the game.
    pub actions:Vec<String>,
}
impl Default for ModSettings {fn default()->Self{Self{enabled:true,game_id:String::new(),game_api:1,profile:"default".into(),override_paths:Vec::new(),actions:Vec::new()}}}
#[derive(Clone,Debug,PartialEq,Eq,Serialize,Deserialize)]
pub struct ModDependency {pub id:String,#[serde(default)]pub version:Option<String>}
#[derive(Clone,Debug,Default,PartialEq,Eq,Serialize,Deserialize)]
#[serde(default)]
pub struct ModManifest {
    /// Canonical UUID v4, generated once by `new` and retained across releases.
    pub id:String,pub name:String,pub author:String,pub version:String,
    pub game_id:String,pub game_api:u32,
    pub dependencies:Vec<ModDependency>,pub conflicts:Vec<String>,pub load_after:Vec<String>,
    /// Base-game path -> packaged replacement path. Later enabled mods win.
    pub overrides:BTreeMap<String,String>,
    /// Base-game scene -> packaged additive scene. Entries receive stable namespaced ids.
    pub additions:BTreeMap<String,String>,
    pub actions:Vec<String>,
}
#[derive(Clone,Debug,Default,PartialEq,Eq,Serialize,Deserialize)]
#[serde(default)]
pub struct ModProfile {pub enabled:Vec<String>}
#[derive(Clone,Debug,PartialEq,Eq,Serialize,Deserialize)]
pub struct LockedMod {pub id:String,pub version:String,pub content_hash:String}
#[derive(Clone,Debug,Default,PartialEq,Eq,Serialize,Deserialize)]
#[serde(default)]
pub struct ModLock {pub game_id:String,pub game_api:u32,pub mods:Vec<LockedMod>}
impl ModLock {
    pub fn check(&self,required:&Self)->Result<(),String>{
        if self==required{return Ok(());}
        if self.game_id!=required.game_id||self.game_api!=required.game_api{return Err("The save/session targets a different game or mod API".into());}
        let wanted=required.mods.iter().map(|m|format!("{}@{} [{}]",m.id,m.version,m.content_hash.chars().take(12).collect::<String>())).collect::<Vec<_>>().join(", ");
        Err(format!("Mod profile mismatch (versions, contents and order must match). Required: [{}]",wanted))
    }
}
pub fn valid_id(s:&str)->bool{!s.is_empty()&&s.len()<=96&&s.bytes().all(|c|c.is_ascii_alphanumeric()||matches!(c,b'.'|b'_'|b'-'))&&s!="."&&s!=".."}
pub fn valid_mod_id(s:&str)->bool{
    s.parse::<crate::ObjectId>().is_ok_and(|id|{let b=id.raw().to_be_bytes();id.to_string()==s&&(b[6]>>4)==4&&(b[8]&0xc0)==0x80})
}
pub fn relative_asset(s:&str)->bool{
    !s.is_empty()&&s.len()<=1024&&!s.contains(['\\',':','\0'])&&!s.starts_with('/')&&s.split('/').all(|p|!p.is_empty()&&p!="."&&p!="..")
}
impl ModManifest {
    /// Authoring only: deserialization and packaging must never regenerate identity.
    pub fn new(game_id:impl Into<String>,game_api:u32)->Self{
        Self{id:crate::ObjectId::new().to_string(),name:"New mod".into(),version:"1.0.0".into(),game_id:game_id.into(),game_api,..Default::default()}
    }
    pub fn validate(&self)->Result<(),String>{
        if !valid_mod_id(&self.id){return Err("Mod ID must be a canonical UUID v4, created once and retained across updates".into());}
        if !valid_id(&self.game_id){return Err("Game ID must use letters, digits, dot, dash or underscore".into());}
        if self.game_api==0||self.version.trim().is_empty()||self.version.len()>64{return Err("A mod needs a version and a nonzero game API".into());}
        if self.dependencies.len()+self.conflicts.len()+self.load_after.len()>256||self.overrides.len()+self.additions.len()>4096||self.actions.len()>256{return Err("Mod manifest exceeds its entry limit".into());}
        for id in self.dependencies.iter().map(|d|&d.id).chain(&self.conflicts).chain(&self.load_after){if !valid_mod_id(id)||id==&self.id{return Err(format!("Invalid/self dependency or conflict: {id}"));}}
        for (target,source)in self.overrides.iter().chain(&self.additions){if !relative_asset(target)||!relative_asset(source){return Err(format!("Invalid mod asset path: {target} -> {source}"));}}
        if self.additions.iter().any(|(a,b)|!a.ends_with(".scene")||!b.ends_with(".scene")){return Err("Scene additions must reference .scene files".into());}
        if self.actions.iter().any(|a|a.is_empty()||a.len()>256||!a.contains('.')){return Err("Actions must be Component.Action names".into());}
        Ok(())
    }
    pub fn check_game(&self,settings:&ModSettings)->Result<(),String>{
        self.validate()?;
        if !settings.enabled{return Err("This game disables mods".into());}
        if self.game_id!=settings.game_id||self.game_api!=settings.game_api{return Err(format!("{} requires game {} API {}",self.id,self.game_id,self.game_api));}
        for path in self.overrides.keys().chain(self.additions.keys()){
            if !settings.override_paths.iter().any(|p|p==path||(p.ends_with('/')&&path.starts_with(p))){return Err(format!("{}: game does not expose {path}",self.id));}
        }
        for action in &self.actions{if !settings.actions.contains(action){return Err(format!("{}: game does not expose action {action}",self.id));}}
        Ok(())
    }
}
/// Stable topological sort. Profile order decides ties and conflicting overrides.
/// Dependencies must be explicitly enabled; missing mods never silently disappear.
pub fn resolve<'a>(profile:&ModProfile,installed:&'a BTreeMap<String,ModManifest>,settings:&ModSettings)->Result<Vec<&'a ModManifest>,String>{
    if profile.enabled.len()>256{return Err("At most 256 mods may be enabled".into());}
    let selected:BTreeSet<_>=profile.enabled.iter().cloned().collect();
    if selected.len()!=profile.enabled.len(){return Err("A mod occurs more than once in the profile".into());}
    for id in &profile.enabled{
        let m=installed.get(id).ok_or_else(||format!("Missing installed mod: {id}"))?;m.check_game(settings)?;
        if &m.id!=id{return Err("Installed mod ID does not match its manifest".into());}
        for d in &m.dependencies{
            let dep=installed.get(&d.id).filter(|_|selected.contains(&d.id)).ok_or_else(||format!("{id} requires enabled mod {}",d.id))?;
            if d.version.as_ref().is_some_and(|v|v!=&dep.version){return Err(format!("{id} requires {} version {}",d.id,d.version.as_deref().unwrap()));}
        }
        if let Some(conflict)=m.conflicts.iter().find(|c|selected.contains(*c)){return Err(format!("{id} conflicts with {conflict}"));}
    }
    let mut ordered=Vec::new();let mut done=BTreeSet::new();
    while ordered.len()<selected.len(){
        let next=profile.enabled.iter().filter(|id|!done.contains(*id)).find(|id|{
            let m=&installed[*id];m.dependencies.iter().all(|d|done.contains(&d.id))&&m.load_after.iter().filter(|id|selected.contains(*id)).all(|id|done.contains(id))
        }).ok_or("Mod dependency/load-order cycle")?;
        ordered.push(&installed[next]);done.insert(next.clone());
    }
    Ok(ordered)
}

#[cfg(test)]mod tests{
    use super::*;
    #[test]fn mod_resolution_orders_dependencies_and_rejects_invalid_profiles(){
        let settings=ModSettings{game_id:"test".into(),..Default::default()};let a=ModManifest::new("test",1);let mut b=ModManifest::new("test",1);
        let (aid,bid)=(a.id.clone(),b.id.clone());b.dependencies.push(ModDependency{id:aid.clone(),version:Some("1.0.0".into())});
        let mut all=BTreeMap::from([(aid.clone(),a),(bid.clone(),b)]);let profile=ModProfile{enabled:vec![bid.clone(),aid.clone()]};
        assert_eq!(resolve(&profile,&all,&settings).unwrap().iter().map(|m|m.id.as_str()).collect::<Vec<_>>(),[aid.as_str(),bid.as_str()]);
        assert!(resolve(&ModProfile{enabled:vec![bid.clone()]},&all,&settings).is_err());
        all.get_mut(&aid).unwrap().load_after.push(bid.clone());assert!(resolve(&profile,&all,&settings).is_err());
        all.get_mut(&aid).unwrap().load_after.clear();all.get_mut(&aid).unwrap().conflicts.push(bid);assert!(resolve(&profile,&all,&settings).is_err());
    }
    #[test]fn mod_permissions_and_save_locks_fail_closed(){
        let mut manifest=ModManifest::new("test",1);manifest.overrides.insert("scenes/main.scene".into(),"custom.scene".into());let mut settings=ModSettings{game_id:"test".into(),..Default::default()};
        assert!(manifest.check_game(&settings).is_err());settings.override_paths.push("scenes/".into());assert!(manifest.check_game(&settings).is_ok());
        for bad in ["../outside","/absolute","a/../b","C:/file",r"a\b"]{assert!(!relative_asset(bad));}
        let a=LockedMod{id:manifest.id,version:"1".into(),content_hash:"abc".into()};let mut required=ModLock{game_id:"test".into(),game_api:1,mods:vec![a]};let current=required.clone();assert!(current.check(&required).is_ok());required.mods[0].content_hash="def".into();assert!(current.check(&required).is_err());
    }
    #[test]fn mod_uuid_is_generated_only_at_creation_and_survives_releases(){
        let mut m=ModManifest::new("test",1);let id=m.id.clone();assert!(valid_mod_id(&id));assert_ne!(ModManifest::new("test",1).id,id);
        m.version="2.0.0".into();let decoded:ModManifest=ron::from_str(&ron::to_string(&m).unwrap()).unwrap();assert_eq!(decoded.id,id);assert!(decoded.validate().is_ok());
        for bad in ["weapons","00000000-0000-0000-0000-000000000000","00000000-0000-5000-8000-000000000001"]{assert!(!valid_mod_id(bad));}
        let missing:ModManifest=ron::from_str("(game_id:\"test\",game_api:1,version:\"1\")").unwrap();assert!(missing.validate().is_err());
    }
}
