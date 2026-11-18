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
    /// Size of the live allocation, including alignment; zero for a taken or destroyed buffer.
    pub(crate) fn allocated_bytes(&self) -> u64 {
        self.allocation.as_ref().map_or(0, |a| a.size())
    }

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
    let mut map = QUEUE_LOCKS.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner());
    map.entry(queue.as_raw()).or_default().clone()
}

type QueueLocks = std::sync::Mutex<std::collections::HashMap<u64, std::sync::Arc<std::sync::Mutex<()>>>>;

static QUEUE_LOCKS: std::sync::OnceLock<QueueLocks> = std::sync::OnceLock::new();

pub(crate) fn with_queues_locked<R>(run: impl FnOnce() -> R) -> R {
    let locks: Vec<std::sync::Arc<std::sync::Mutex<()>>> = {
        let map = QUEUE_LOCKS.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner());
        let mut keyed: Vec<(u64, std::sync::Arc<std::sync::Mutex<()>>)> = map.iter().map(|(key, lock)| (*key, lock.clone())).collect();
        keyed.sort_by_key(|entry| entry.0);
        keyed.into_iter().map(|entry| entry.1).collect()
    };
    let guards: Vec<std::sync::MutexGuard<'_, ()>> = locks.iter().map(|lock| lock.lock().unwrap_or_else(|e| e.into_inner())).collect();
    let out = run();
    drop(guards);
    out
}

#[derive(Default)]
pub(crate) struct DeferredUploads {
    pending: Vec<DeferredUpload>,
}

struct DeferredUpload {
    pool: vk::CommandPool,
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    buffers: Vec<Buffer>,
    textures: Vec<crate::texture::GpuTexture>,
}

impl DeferredUpload {
    fn release(self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_fence(self.fence, None);
            device.free_command_buffers(self.pool, &[self.cb]);
        }
        for mut buffer in self.buffers {
            buffer.destroy(device, allocator);
        }
        for mut texture in self.textures {
            texture.destroy(device, allocator);
        }
    }
}

impl DeferredUploads {
    pub(crate) fn submit(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        pool: vk::CommandPool,
        queue: vk::Queue,
        mut staging: Buffer,
        record: impl FnOnce(vk::CommandBuffer),
    ) -> Result<()> {
        let submitted = (|| -> Result<(vk::CommandBuffer, vk::Fence)> {
            unsafe {
                let alloc = vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                let cb = device.allocate_command_buffers(&alloc)?[0];
                let recorded = (|| -> Result<vk::Fence> {
                    device.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
                    record(cb);
                    device.end_command_buffer(cb)?;
                    let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
                    let cbs = [cb];
                    let submit = vk::SubmitInfo::default().command_buffers(&cbs);
                    let lock = queue_lock(queue);
                    let result = {
                        let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
                        device.queue_submit(queue, &[submit], fence)
                    };
                    if let Err(e) = result {
                        device.destroy_fence(fence, None);
                        return Err(e.into());
                    }
                    Ok(fence)
                })();
                match recorded {
                    Ok(fence) => Ok((cb, fence)),
                    Err(e) => {
                        device.free_command_buffers(pool, &[cb]);
                        Err(e)
                    }
                }
            }
        })();
        match submitted {
            Ok((cb, fence)) => {
                self.pending.push(DeferredUpload { pool, cb, fence, buffers: vec![staging], textures: Vec::new() });
                Ok(())
            }
            Err(e) => {
                staging.destroy(device, allocator);
                Err(e)
            }
        }
    }

    pub(crate) fn retire_texture(&mut self, device: &ash::Device, allocator: &mut Allocator, mut texture: crate::texture::GpuTexture) {
        match self.pending.last_mut() {
            Some(last) => last.textures.push(texture),
            None => texture.destroy(device, allocator),
        }
    }

    pub(crate) fn reap(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        let done = self.pending.iter().take_while(|p| unsafe { device.get_fence_status(p.fence) }.unwrap_or(false)).count();
        for entry in self.pending.drain(..done) {
            entry.release(device, allocator);
        }
    }

    pub(crate) fn finish(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for entry in self.pending.drain(..) {
            unsafe {
                let _ = device.wait_fences("wait:fence@deferred_uploads", &[entry.fence], true, u64::MAX);
            }
            entry.release(device, allocator);
        }
    }
}

pub(crate) fn submit_upload(
    device: &ash::Device,
    allocator: &mut Allocator,
    pool: vk::CommandPool,
    queue: vk::Queue,
    deferred: Option<&mut DeferredUploads>,
    mut staging: Buffer,
    record: impl FnOnce(vk::CommandBuffer),
) -> Result<()> {
    match deferred {
        Some(uploads) => uploads.submit(device, allocator, pool, queue, staging, record),
        None => {
            let result = one_time_submit(device, pool, queue, record);
            staging.destroy(device, allocator);
            result
        }
    }
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

#[cfg(test)]
mod queue_lock_tests {
    use super::*;

    #[test]
    fn an_idle_wait_keeps_one_time_submits_off_the_queues_until_it_ends() {
        use ash::vk::Handle as _;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        let queue = vk::Queue::from_raw(0xfeed_0001);
        let other = vk::Queue::from_raw(0xfeed_0002);
        let (lock, _other_lock) = (queue_lock(queue), queue_lock(other));
        let idle_done = Arc::new(AtomicBool::new(false));
        let submitted_during_idle = Arc::new(AtomicBool::new(false));
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let worker = {
            let (idle_done, submitted_during_idle) = (idle_done.clone(), submitted_during_idle.clone());
            std::thread::spawn(move || {
                entered_rx.recv().unwrap();
                let _guard = lock.lock().unwrap();
                if !idle_done.load(Ordering::SeqCst) {
                    submitted_during_idle.store(true, Ordering::SeqCst);
                }
            })
        };
        with_queues_locked(|| {
            entered_tx.send(()).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(60));
            idle_done.store(true, Ordering::SeqCst);
        });
        worker.join().unwrap();
        assert!(!submitted_during_idle.load(Ordering::SeqCst), "a submit ran while the device was being idled");
        assert_eq!(with_queues_locked(|| 7), 7, "the locks are released afterwards");
        let again = queue_lock(queue);
        assert!(again.try_lock().is_ok());
    }
}
