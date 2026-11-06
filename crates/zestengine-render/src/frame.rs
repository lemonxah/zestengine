//! Per-frame-in-flight resources: command buffer and sync primitives.

use anyhow::Result;
use ash::vk;

pub const FRAMES_IN_FLIGHT: usize = 2;

/// GPU timestamp query slots per frame: two per zone (begin/end), zone Z using
/// slots `2*Z` and `2*Z+1`. Zone indices are in `gpu_zone` (lib.rs). Read back
/// with availability so zones not recorded this frame simply report 0.
pub const TS_COUNT: u32 = 62;

pub struct Frame {
    pub command_buffer: vk::CommandBuffer,
    /// Signaled by acquire, waited on by the render submit.
    pub image_available: vk::Semaphore,
    /// Signaled by the render submit, waited on by the CPU before reusing
    /// this frame's resources.
    pub in_flight: vk::Fence,
    /// Per-frame GPU timestamp pool (TS_COUNT queries). Read back once this
    /// frame slot's fence has retired, giving GPU-side pass timings.
    pub timestamps: vk::QueryPool,
}

impl Frame {
    pub fn new(device: &ash::Device, command_buffer: vk::CommandBuffer) -> Result<Self> {
        let image_available =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? };
        let in_flight = unsafe {
            device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )?
        };
        let timestamps = unsafe {
            device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(TS_COUNT),
                None,
            )?
        };
        Ok(Self {
            command_buffer,
            image_available,
            in_flight,
            timestamps,
        })
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_semaphore(self.image_available, None);
            device.destroy_fence(self.in_flight, None);
            device.destroy_query_pool(self.timestamps, None);
        }
    }
}
