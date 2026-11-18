//! Bounded, asynchronous uploads into a persistent scalar atlas. A residency
//! owner must retain page leases until all snapshots using them have retired;
//! queue ordering alone cannot prevent an old snapshot reading a reused page.
use super::*;

pub(crate) struct VolumePage<'a> {
    pub origin: [u32; 3],
    pub size: [u32; 3],
    pub data: &'a [f32],
}

struct Upload {
    staging: Buffer,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    bytes: u64,
}

pub(crate) struct VolumePages {
    texture: GpuTexture,
    dims: [u32; 3],
    pool: vk::CommandPool,
    queue: vk::Queue,
    initialized: bool,
    pending: Vec<Upload>,
    staging_limit: u64,
    staging_bytes: u64,
}

fn range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1)
}

/// Validate before allocating or touching the image. Overlapping writes in one
/// copy command are forbidden; callers must coalesce repeated page requests.
fn layout(dims: [u32; 3], pages: &[VolumePage<'_>], limit: u64)
    -> Result<(u64, Vec<vk::BufferImageCopy>)>
{
    let mut bytes = 0u64;
    let mut copies = Vec::with_capacity(pages.len());
    for (i, p) in pages.iter().enumerate() {
        anyhow::ensure!((0..3).all(|a| p.size[a] > 0
            && p.origin[a].checked_add(p.size[a]).is_some_and(|end| end <= dims[a])),
            "volume page is outside its atlas");
        let count = p.size.iter().try_fold(1usize, |n, &s| n.checked_mul(s as usize));
        anyhow::ensure!(count == Some(p.data.len()), "volume page data length mismatch");
        for other in &pages[..i] {
            anyhow::ensure!((0..3).any(|a| p.origin[a] >= other.origin[a] + other.size[a]
                || other.origin[a] >= p.origin[a] + p.size[a]), "overlapping volume page writes");
        }
        copies.push(vk::BufferImageCopy::default().buffer_offset(bytes)
            .image_subresource(vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
            .image_offset(vk::Offset3D { x: p.origin[0] as i32, y: p.origin[1] as i32, z: p.origin[2] as i32 })
            .image_extent(vk::Extent3D { width: p.size[0], height: p.size[1], depth: p.size[2] }));
        bytes = bytes.checked_add((p.data.len() as u64).checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("volume page byte count overflow"))?)
            .ok_or_else(|| anyhow::anyhow!("volume upload byte count overflow"))?;
        anyhow::ensure!(bytes <= limit, "volume upload exceeds staging budget; split the batch");
    }
    Ok((bytes, copies))
}

impl VolumePages {
    /// The queue must be the graphics queue that samples this atlas. Use a
    /// private pool so frame-pool resets cannot invalidate pending uploads.
    /// No initial staging allocation, submission, or fence wait is performed.
    pub fn new(device: &ash::Device, allocator: &mut Allocator,
        queue: vk::Queue, family: u32, dims: [u32; 3], staging_limit: u64) -> Result<Self>
    {
        anyhow::ensure!(dims.iter().all(|&n| n > 0 && n <= i32::MAX as u32), "invalid volume atlas dimensions");
        anyhow::ensure!(staging_limit >= 4, "invalid volume staging budget");
        let image = unsafe { device.create_image(&vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_3D).format(vk::Format::R32_SFLOAT)
            .extent(vk::Extent3D { width: dims[0], height: dims[1], depth: dims[2] })
            .mip_levels(1).array_layers(1).samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC), None)? };
        let allocation = match allocator.allocate(&AllocationCreateDesc {
            name: "mesh distance field pages", requirements: unsafe { device.get_image_memory_requirements(image) },
            location: MemoryLocation::GpuOnly, linear: false, allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }) {
            Ok(a) => a,
            Err(e) => { unsafe { device.destroy_image(image, None) }; return Err(e.into()); }
        };
        let mut texture = GpuTexture { image, view: vk::ImageView::null(), allocation: Some(allocation) };
        let result = (|| -> Result<vk::CommandPool> {
            let a = texture.allocation.as_ref().unwrap();
            unsafe {
                device.bind_image_memory(image, a.memory(), a.offset())?;
                texture.view = device.create_image_view(&vk::ImageViewCreateInfo::default()
                    .image(image).view_type(vk::ImageViewType::TYPE_3D).format(vk::Format::R32_SFLOAT)
                    .subresource_range(range()), None)?;
                Ok(device.create_command_pool(&vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family).flags(vk::CommandPoolCreateFlags::TRANSIENT), None)?)
            }
        })();
        let pool = match result { Ok(p) => p, Err(e) => { texture.destroy(device, allocator); return Err(e); } };
        Ok(Self { texture, dims, pool, queue, initialized: false, pending: Vec::new(), staging_limit, staging_bytes: 0 })
    }

    /// Ready for *subsequent submissions on this queue* after the first accepted
    /// upload. Only explicitly uploaded pages may be sampled; the rest is undefined.
    pub fn texture(&self) -> Option<&GpuTexture> { self.initialized.then_some(&self.texture) }

    pub fn staging_bytes(&self) -> u64 { self.staging_bytes }

    /// Reap completed uploads without waiting. Allocations include Vulkan's
    /// alignment, so the staging budget measures actual retained buffer memory.
    pub fn collect(&mut self, device: &ash::Device, allocator: &mut Allocator) -> Result<()> {
        let mut i = 0;
        while i < self.pending.len() {
            if unsafe { device.get_fence_status(self.pending[i].fence)? } {
                let mut upload = self.pending.swap_remove(i);
                unsafe {
                    device.destroy_fence(upload.fence, None);
                    device.free_command_buffers(self.pool, &[upload.command]);
                }
                upload.staging.destroy(device, allocator);
                self.staging_bytes -= upload.bytes;
            } else { i += 1; }
        }
        Ok(())
    }

    /// False means budget pressure: nothing was submitted or changed; retry
    /// later. True publishes the entire batch in queue order, without a CPU wait.
    /// Calls and graphics submissions must be externally ordered on this queue.
    pub fn upload(&mut self, device: &ash::Device, allocator: &mut Allocator, pages: &[VolumePage<'_>]) -> Result<bool> {
        let (bytes, copies) = layout(self.dims, pages, self.staging_limit)?;
        self.collect(device, allocator)?;
        if bytes == 0 { return Ok(true); }
        if self.pending.len() >= 4 || bytes > self.staging_limit - self.staging_bytes { return Ok(false); }
        let mut staging = Buffer::new(device, allocator, bytes, vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu, "mesh field page upload")?;
        let allocated = staging.allocated_bytes();
        if allocated > self.staging_limit {
            staging.destroy(device, allocator);
            anyhow::bail!("aligned volume upload exceeds staging budget; split the batch");
        }
        if allocated > self.staging_limit - self.staging_bytes {
            staging.destroy(device, allocator);
            return Ok(false);
        }
        for (p, c) in pages.iter().zip(&copies) { staging.write(c.buffer_offset as usize, bytemuck::cast_slice(p.data)); }
        let mut command = vk::CommandBuffer::null();
        let mut fence = vk::Fence::null();
        let result = (|| -> Result<()> { unsafe {
            command = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default()
                .command_pool(self.pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))?[0];
            device.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            let begin = vk::ImageMemoryBarrier2::default().image(self.texture.image).subresource_range(range())
                .old_layout(if self.initialized { vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL } else { vk::ImageLayout::UNDEFINED })
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER).dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE);
            crate::sync_check::barrier2(&device, command, &vk::DependencyInfo::default().image_memory_barriers(&[begin]));
            device.cmd_copy_buffer_to_image(command, staging.handle, self.texture.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &copies);
            let end = vk::ImageMemoryBarrier2::default().image(self.texture.image).subresource_range(range())
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL).new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER).src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS).dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ);
            crate::sync_check::barrier2(&device, command, &vk::DependencyInfo::default().image_memory_barriers(&[end]));
            device.end_command_buffer(command)?;
            fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let lock = crate::alloc::queue_lock(self.queue);
            let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
            device.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&[command])], fence)?;
            Ok(())
        } })();
        if let Err(e) = result {
            unsafe {
                device.destroy_fence(fence, None);
                if command != vk::CommandBuffer::null() { device.free_command_buffers(self.pool, &[command]); }
            }
            staging.destroy(device, allocator);
            return Err(e);
        }
        self.initialized = true;
        self.staging_bytes += allocated;
        self.pending.push(Upload { staging, command, fence, bytes: allocated });
        Ok(true)
    }

    /// Caller must retire every sampling frame first. Teardown may wait for
    /// this atlas's uploads, but normal upload/collection never waits on fences.
    pub fn destroy(mut self, device: &ash::Device, allocator: &mut Allocator) -> Result<()> {
        let fences: Vec<_> = self.pending.iter().map(|u| u.fence).collect();
        if !fences.is_empty() { unsafe { device.wait_for_fences(&fences, true, u64::MAX)?; } }
        self.collect(device, allocator)?;
        unsafe { device.destroy_command_pool(self.pool, None); }
        self.texture.destroy(device, allocator);
        Ok(())
    }
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn page_upload_validation_rejects_invalid_or_ambiguous_batches() {
        let data = [1.; 8];
        let page = |origin, size| VolumePage { origin, size, data: &data };
        let pages = [page([0; 3], [2; 3]), page([2, 0, 0], [2; 3])];
        let (bytes, copies) = layout([4; 3], &pages, 64).unwrap();
        assert_eq!(bytes, 64); assert_eq!(copies[1].buffer_offset, 32);
        assert!(layout([4; 3], &pages, 63).is_err());
        assert!(layout([4; 3], &[page([0; 3], [2; 3]), page([1; 3], [2; 3])], 64).is_err());
        for (origin, size) in [([3; 3], [2; 3]), ([u32::MAX; 3], [2; 3]), ([0; 3], [0; 3]), ([0; 3], [1; 3])] {
            assert!(layout([4; 3], &[page(origin, size)], 64).is_err());
        }
    }
}
