//! Mod mounts are scoped to Play; Stop restores the authored project and scene.
use super::*;
impl EngineApp{
    pub(super) fn mount_mods_for_play(&mut self)->anyhow::Result<bool>{
        let Some(mount)=zestengine_engine::modding::prepare(&self.project_root,&self.components)?else{return Ok(false);};
        anyhow::ensure!(!self.scene_dirty,"Save the scene before starting Play with mods");
        anyhow::ensure!(self.net.as_ref().is_none_or(|net|net.matches_mods(&mount.lock)),"Disconnect before changing the multiplayer mod profile");
        let origin=self.current_scene_path.clone().ok_or_else(||anyhow::anyhow!("Save the scene before starting Play with mods"))?;
        let rel=origin.strip_prefix(&self.project_root)?;
        let uploader=self.renderer.as_ref().and_then(|r|r.uploader());
        let data=zestengine_engine::scene_loader::load_on_worker(&mount.root.join(rel),&mount.root,uploader.as_ref(),&[],&|_|{}).map_err(anyhow::Error::msg)?;
        let (scene,post)=zestengine_engine::scene_loader::install(data,self.renderer.as_mut(),&self.components,&mut self.shaders,&mount.root,|_|{})?;
        let original=std::mem::replace(&mut self.project_root,mount.root.clone());
        self.scene=scene;self.post_load=Some(post);self.mod_mount=Some((original,mount));self.play_origin_scene=Some(origin);
        Ok(true)
    }
    pub(super) fn unmount_play_mods(&mut self){
        if let Some((root,mount))=self.mod_mount.take(){
            // A connected session cannot outlive the content it was admitted with.
            self.net=None;self.local_server=None;self.project_root=root;drop(mount);
        }
    }
}
