//! Buffer allocation and upload helpers on top of gpu-allocator.

use crate::hitch::WaitNamed as _;
use anyhow::{Context as _, Result};
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

pub(crate) struct Buffer {
    pub handle: vk::Buffer,
    allocation: Option<Allocation>,
}

impl Buffer {
    pub fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
    ) -> Result<Self> {
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let handle = unsafe { device.create_buffer(&info, None)? };
        let requirements = unsafe { device.get_buffer_memory_requirements(handle) };
        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .with_context(|| format!("allocating buffer memory for {name}"))?;
        unsafe { device.bind_buffer_memory(handle, allocation.memory(), allocation.offset())? };
        Ok(Self {
            handle,
            allocation: Some(allocation),
        })
    }

    /// GPU virtual address (buffer must be created with SHADER_DEVICE_ADDRESS
    /// usage and the device's bufferDeviceAddress feature enabled).
    pub fn device_address(&self, device: &ash::Device) -> vk::DeviceAddress {
        unsafe {
            device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(self.handle),
            )
        }
    }

    /// Read the whole buffer back to host memory (host-visible buffers only).
    pub fn read(&self) -> Vec<u8> {
        self.allocation
            .as_ref()
            .and_then(|a| a.mapped_slice())
            .map(|s| s.to_vec())
            .unwrap_or_default()
    }

    /// The mapped contents of a host-visible buffer.
    pub fn mapped(&self) -> Option<&[u8]> {
        self.allocation.as_ref().and_then(|a| a.mapped_slice())
    }

    /// Write into a host-visible buffer.
    pub fn write(&mut self, offset: usize, data: &[u8]) {
        let mapped = self
            .allocation
            .as_mut()
            .and_then(|a| a.mapped_slice_mut())
            .expect("buffer is not host-visible");
        mapped[offset..offset + data.len()].copy_from_slice(data);
    }

    /// Move the GPU object out into a new `Buffer`, leaving a null husk that
    /// `destroy`s as a no-op (deferred frees own the real one).
    pub fn take(&mut self) -> Buffer {
        Buffer { handle: std::mem::take(&mut self.handle), allocation: self.allocation.take() }
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe { device.destroy_buffer(self.handle, None) };
        if let Some(allocation) = self.allocation.take() {
            let _ = allocator.free(allocation);
        }
    }
}

/// The lock every one-time submit takes around `vkQueueSubmit` on `queue`
/// (a VkQueue is externally synchronised). The background `Uploader` and the
/// main thread both submit one-time uploads on the transfer queue; they share
/// this lock instead of each owning the queue. Keyed by the queue handle.
pub(crate) fn queue_lock(queue: vk::Queue) -> std::sync::Arc<std::sync::Mutex<()>> {
    use ash::vk::Handle as _;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};
    static LOCKS: OnceLock<Mutex<HashMap<u64, Arc<Mutex<()>>>>> = OnceLock::new();
    let mut map = LOCKS.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner());
    map.entry(queue.as_raw()).or_default().clone()
}

/// Record and submit a one-time command buffer, waiting for completion.
/// Used for uploads; not on any hot path. The submit holds the queue's shared
/// lock ([`queue_lock`]); the fence wait does not.
pub(crate) fn one_time_submit(
    device: &ash::Device,
    pool: vk::CommandPool,
    queue: vk::Queue,
    record: impl FnOnce(vk::CommandBuffer),
) -> Result<()> {
    unsafe {
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cb = device.allocate_command_buffers(&alloc)?[0];
        device.begin_command_buffer(
            cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        record(cb);
        device.end_command_buffer(cb)?;

        let cbs = [cb];
        let submit = vk::SubmitInfo::default().command_buffers(&cbs);
        let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        let lock = queue_lock(queue);
        let submitted = {
            let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
            device.queue_submit(queue, &[submit], fence)
        };
        let result = submitted
            .and_then(|()| device.wait_fences("wait:fence@one_time_submit", &[fence], true, u64::MAX));
        device.destroy_fence(fence, None);
        device.free_command_buffers(pool, &cbs);
        result?;
    }
    Ok(())
}

/// Upload `data` into a new device-local buffer via a staging buffer.
pub(crate) fn upload_buffer(
    device: &ash::Device,
    allocator: &mut Allocator,
    pool: vk::CommandPool,
    queue: vk::Queue,
    data: &[u8],
    usage: vk::BufferUsageFlags,
    name: &str,
) -> Result<Buffer> {
    let size = data.len() as u64;
    let mut staging = Buffer::new(
        device,
        allocator,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
        "staging",
    )?;
    staging.write(0, data);

    let dst = Buffer::new(
        device,
        allocator,
        size,
        usage | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuOnly,
        name,
    )?;

    one_time_submit(device, pool, queue, |cb| {
        let region = vk::BufferCopy::default().size(size);
        unsafe { device.cmd_copy_buffer(cb, staging.handle, dst.handle, &[region]) };
    })?;

    staging.destroy(device, allocator);
    Ok(dst)
}
