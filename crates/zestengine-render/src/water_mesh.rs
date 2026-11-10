//! Water topology replacement keeps handles stable as zone tiles change LOD.
use crate::{MeshData,MeshHandle,Renderer};
use anyhow::{Result,ensure};

impl Renderer {
    pub fn replace_water_mesh(&mut self,handle:MeshHandle,data:&MeshData)->Result<()> {
        ensure!(handle.0<self.meshes.len(),"invalid water mesh handle");
        ensure!(!data.vertices.is_empty() && !data.indices.is_empty(),"empty water mesh");
        // The index upload submits to the graphics queue and waits on its
        // fence. All earlier draws using the old buffers have retired before
        // they are freed. Do not use this for per-frame wave displacement.
        let uploaded=self.upload_mesh_skinned(data)?;
        debug_assert_eq!(uploaded.0,self.meshes.len()-1);
        let replacement=self.meshes.pop().unwrap();
        let mut old=std::mem::replace(&mut self.meshes[handle.0],replacement);
        let allocator=self.allocator();
        let mut alloc=allocator.lock().unwrap();
        old.vertex_buffer.destroy(&self.ctx.device,&mut alloc);
        old.index_buffer.destroy(&self.ctx.device,&mut alloc);
        drop(alloc);
        self.bump_mesh_version(handle);
        Ok(())
    }
}
