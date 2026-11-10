//! Per-view spatial final-gather quality. Lumen halves screen-probe spacing at
//! quality 6; this operates on Flux's existing grid and directional layout.
use super::*;

pub(super) fn probe_extent(extent:vk::Extent2D,quality:Option<f32>,spacing:u32,max_buffer:u32)->vk::Extent2D {
    let quality=quality.filter(|v|v.is_finite()).unwrap_or(1.0);
    let spacing=spacing.clamp(2,32);
    let mut div=if quality>=6.0 {(spacing/2).max(1)} else {spacing};
    loop {
        let size=vk::Extent2D {width:extent.width.div_ceil(div).max(1),height:extent.height.div_ceil(div).max(1)};
        let probes=u64::from(size.width)*u64::from(size.height);
        // Raw adaptive rays and directional history are storage buffers. Do
        // not let high-resolution screenshots exceed the device's range.
        if probes*gpu_gi::SCREEN_PROBE_BYTES.max(17*16)+16<=u64::from(max_buffer)
            || (size.width==1 && size.height==1) {return size;}
        div=div.saturating_mul(2);
    }
}

impl ScreenGiTargets {
    fn replace_quality(&mut self,ctx:&GpuContext,alloc:&mut Allocator,quality:Option<f32>,spacing:u32)->Result<Option<Self>> {
        let extent=self.full_history[0].extent;
        if self.probe_extent==probe_extent(extent,quality,spacing,ctx.caps.max_storage_buffer_range) {return Ok(None);}
        let next=Self::new_with_gather(ctx,alloc,extent,quality,spacing)?;
        Ok(Some(std::mem::replace(self,next)))
    }
}

impl Renderer {
    pub(super) fn prepare_main_gather_quality(&mut self,input:&FrameInput)->Result<()> {
        if (!input.render_viewport && input.viewport_extent.is_none())
            || !self.gi_has_gdf()
            || !self.tier.realtime_gi()
        {return Ok(());}
        let Some(allocator)=self.allocator.clone() else {return Ok(());};
        let target=if input.viewport_extent.is_some() {
            let Some(view)=self.viewport_target.as_mut() else {return Ok(());};
            &mut view.sgi
        } else {&mut self.sgi};
        if let Some(old)=target.replace_quality(&self.ctx,&mut allocator.lock().unwrap(),input.postfx.flux.final_gather_quality,self.flux_settings.probe_spacing)? {
            self.retired_gi_targets.push((old,FRAMES_IN_FLIGHT+1));
            self.sgi_history_valid=false;self.sgi_parity=0;self.sgi_last_view_proj=None;
        }
        Ok(())
    }
    pub(super) fn prepare_preview_gather_quality(&mut self,quality:Option<f32>)->Result<()> {
        if !self.gi_has_gdf() || !self.tier.realtime_gi() {return Ok(());}
        let Some(allocator)=self.allocator.clone() else {return Ok(());};
        let Some(view)=self.camera_preview.as_mut() else {return Ok(());};
        if let Some(old)=view.sgi.replace_quality(&self.ctx,&mut allocator.lock().unwrap(),quality,self.flux_settings.probe_spacing)? {
            self.retired_gi_targets.push((old,FRAMES_IN_FLIGHT+1));
            self.preview_sgi_history_valid=false;self.preview_sgi_parity=0;self.preview_sgi_last_view_proj=None;
        }
        Ok(())
    }
    pub(super) fn free_retired_gi_targets(&mut self,all:bool) {
        if self.retired_gi_targets.is_empty() {return;}
        let Some(allocator)=self.allocator.clone() else {return;};
        let mut alloc=allocator.lock().unwrap();let mut i=0;
        while i<self.retired_gi_targets.len() {
            self.retired_gi_targets[i].1=self.retired_gi_targets[i].1.saturating_sub(1);
            if all || self.retired_gi_targets[i].1==0 {
                let (mut old,_)=self.retired_gi_targets.swap_remove(i);old.destroy(&self.ctx.device,&mut alloc);
            } else {i+=1;}
        }
    }
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn density_threshold_odd_extents_and_device_buffer_limit() {
        let extent=vk::Extent2D {width:323,height:181};
        for q in [None,Some(1.0),Some(5.99),Some(f32::NAN),Some(f32::INFINITY)] {
            assert_eq!(probe_extent(extent,q,4,u32::MAX),vk::Extent2D {width:81,height:46});
        }
        assert_eq!(probe_extent(extent,Some(6.0),4,u32::MAX),vk::Extent2D {width:162,height:91});
        assert_eq!(probe_extent(extent,None,8,u32::MAX),vk::Extent2D {width:41,height:23});
        assert_eq!(probe_extent(extent,None,16,u32::MAX),vk::Extent2D {width:21,height:12});
        assert_eq!(probe_extent(extent,Some(6.0),8,u32::MAX),vk::Extent2D {width:81,height:46});
        let max_buffer=128*1024*1024;
        let large=probe_extent(vk::Extent2D {width:8192,height:8192},Some(8.0),4,max_buffer);
        assert!(u64::from(large.width)*u64::from(large.height)*gpu_gi::SCREEN_PROBE_BYTES.max(17*16)+16<=u64::from(max_buffer));
    }
}
