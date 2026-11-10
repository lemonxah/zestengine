//! Fenced frame storage for body parameters followed by variable-size
//! footprint data. Shared by desktop, preview/capture and both XR paths.
use std::ops::Deref;
use anyhow::Result;
use ash::vk;
use gpu_allocator::{MemoryLocation, vulkan::Allocator};
use crate::{FrameInput, GpuWater, MAX_WATER_BODIES, alloc::Buffer};

const HEADER: usize = std::mem::size_of::<[GpuWater; MAX_WATER_BODIES]>();

pub(crate) struct WaterSceneBuffer { buffer: Buffer, capacity: usize }

impl Deref for WaterSceneBuffer {
    type Target = Buffer;
    fn deref(&self) -> &Buffer { &self.buffer }
}

impl WaterSceneBuffer {
    pub fn new(device: &ash::Device, allocator: &mut Allocator) -> Result<Self> {
        let mut buffer = Buffer::new(device, allocator, HEADER as u64, vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu, "water scene")?;
        buffer.write(0, bytemuck::cast_slice(&[GpuWater::default(); MAX_WATER_BODIES]));
        Ok(Self { buffer, capacity: HEADER })
    }

    /// Caller has waited for this slot's fence. Rebinding after upload is
    /// mandatory: a growing footprint can replace this slot's allocation.
    pub fn upload(&mut self, device: &ash::Device, allocator: &mut Allocator, input: &FrameInput) -> Result<()> {
        let tail = bytemuck::cast_slice(input.water_coverage);
        let required = HEADER + tail.len();
        if required > self.capacity {
            let capacity = required.next_power_of_two();
            let buffer = Buffer::new(device, allocator, capacity as u64, vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu, "water scene coverage")?;
            self.buffer.destroy(device, allocator);
            self.buffer = buffer;
            self.capacity = capacity;
        }
        self.buffer.write(0, bytemuck::cast_slice(&crate::water_bodies(input)));
        self.buffer.write(HEADER, tail);
        Ok(())
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        self.buffer.destroy(device, allocator);
    }
}
