//! GPU texture upload (RGBA8, no mipmaps yet; M3 adds mip generation).

use anyhow::{Context as _, Result};
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

use crate::alloc::{Buffer, one_time_submit};
use crate::hdr_format::HdrTextureArray;
use crate::types::{CompressedTexture, TexFormat, TextureData};

pub(crate) struct GpuTexture {
    pub image: vk::Image,
    pub view: vk::ImageView,
    allocation: Option<Allocation>,
}

impl GpuTexture {
    /// A color render target that is also sampled (world-space UI). Left in
    /// SHADER_READ_ONLY_OPTIMAL, cleared to transparent.
    pub fn render_target(
        device: &ash::Device,
        allocator: &mut Allocator,
        pool: vk::CommandPool,
        queue: vk::Queue,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Result<Self> {
        Self::cleared(
            device,
            allocator,
            pool,
            queue,
            width,
            height,
            format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
        )
    }

    /// A sampled image with `usage` (plus TRANSFER_DST), cleared to zero and
    /// left in SHADER_READ_ONLY_OPTIMAL.
    #[allow(clippy::too_many_arguments)]
    pub fn cleared(
        device: &ash::Device,
        allocator: &mut Allocator,
        pool: vk::CommandPool,
        queue: vk::Queue,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self> {
        Self::cleared_mips(device, allocator, pool, queue, width, height, format, usage, 1)
    }

    /// [`Self::cleared`] with `levels` mips, every one cleared; the view
    /// covers them all.
    #[allow(clippy::too_many_arguments)]
    pub fn cleared_mips(
        device: &ash::Device,
        allocator: &mut Allocator,
        pool: vk::CommandPool,
        queue: vk::Queue,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        levels: u32,
    ) -> Result<Self> {
        let levels = levels.max(1);
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(levels)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&info, None)? };
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "ui target",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .with_context(|| "allocating GPU memory for ui target")?;
        unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(levels)
            .layer_count(1);
        one_time_submit(device, pool, queue, |cb| unsafe {
            let to_dst = vk::ImageMemoryBarrier2::default()
                .image(image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .subresource_range(range);
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&[to_dst]));
            device.cmd_clear_color_image(
                cb,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue { float32: [0.0; 4] },
                &[range],
            );
            let to_read = vk::ImageMemoryBarrier2::default()
                .image(image)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .subresource_range(range);
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&[to_read]));
        })?;
        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(range),
                None,
            )?
        };
        Ok(Self {
            image,
            view,
            allocation: Some(allocation),
        })
    }

    pub fn upload(
        device: &ash::Device,
        allocator: &mut Allocator,
        pool: vk::CommandPool,
        queue: vk::Queue,
        data: &TextureData,
    ) -> Result<Self> {
        let format = if data.hdr {
            vk::Format::R16G16B16A16_SFLOAT // native HDR float (RGBA f16)
        } else if data.srgb {
            vk::Format::R8G8B8A8_SRGB
        } else {
            vk::Format::R8G8B8A8_UNORM
        };
        Self::upload_pixels(device, allocator, pool, queue, data.width, data.height, format, &data.pixels)
    }

    /// A single-mip 2D image of `format` filled with the raw `pixels` bytes,
    /// left in SHADER_READ_ONLY_OPTIMAL (lookup tables such as the LTC matrices).
    #[allow(clippy::too_many_arguments)]
    pub fn upload_pixels(
        device: &ash::Device,
        allocator: &mut Allocator,
        pool: vk::CommandPool,
        queue: vk::Queue,
        width: u32,
        height: u32,
        format: vk::Format,
        pixels: &[u8],
    ) -> Result<Self> {
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };

        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&info, None)? };
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "texture",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .with_context(|| "allocating GPU memory for texture")?;
        unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };

        let mut staging = Buffer::new(
            device,
            allocator,
            pixels.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "texture staging",
        )?;
        staging.write(0, pixels);

        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);

        one_time_submit(device, pool, queue, |cb| unsafe {
            let to_transfer = vk::ImageMemoryBarrier2::default()
                .image(image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .subresource_range(range);
            let barriers = [to_transfer];
            device.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            );

            let copy = vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_extent(extent);
            device.cmd_copy_buffer_to_image(
                cb,
                staging.handle,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy],
            );

            let to_sampled = vk::ImageMemoryBarrier2::default()
                .image(image)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .subresource_range(range);
            let barriers = [to_sampled];
            device.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            );
        })?;

        staging.destroy(device, allocator);

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(range);
        let view = unsafe { device.create_image_view(&view_info, None)? };

        Ok(Self {
            image,
            view,
            allocation: Some(allocation),
        })
    }

    /// Upload a pre-compressed (BC7/BC6H/ASTC) or raw texture with a full mip chain.
    /// All CPU work (decode, mip generation, block compression) already happened
    /// off the main thread; this just stages each mip and copies it in.
    pub fn upload_compressed(
        device: &ash::Device,
        allocator: &mut Allocator,
        pool: vk::CommandPool,
        queue: vk::Queue,
        tex: &CompressedTexture,
    ) -> Result<Self> {
        Self::upload_compressed_layers(device, allocator, pool, queue, &[tex], false)
    }

    /// [`Self::upload_compressed`] for several same-sized layers of one 2D
    /// array image (atlas pages). `array` selects a 2D-array view even for a
    /// single layer. Every layer must share format, size and mip count.
    pub fn upload_compressed_layers(
        device: &ash::Device,
        allocator: &mut Allocator,
        pool: vk::CommandPool,
        queue: vk::Queue,
        layers: &[&CompressedTexture],
        array: bool,
    ) -> Result<Self> {
        anyhow::ensure!(!layers.is_empty(), "no texture layers to upload");
        let tex = layers[0];
        for l in &layers[1..] {
            anyhow::ensure!(
                l.format == tex.format && l.width == tex.width && l.height == tex.height && l.mips.len() == tex.mips.len(),
                "texture array layers must share format, size and mip count"
            );
        }
        let layer_count = layers.len() as u32;
        let format = match tex.format {
            TexFormat::Bc7Srgb => vk::Format::BC7_SRGB_BLOCK,
            TexFormat::Bc7Unorm => vk::Format::BC7_UNORM_BLOCK,
            TexFormat::Bc6h => vk::Format::BC6H_UFLOAT_BLOCK,
            TexFormat::RgbaSrgb => vk::Format::R8G8B8A8_SRGB,
            TexFormat::RgbaUnorm => vk::Format::R8G8B8A8_UNORM,
            TexFormat::RgbaF16 => vk::Format::R16G16B16A16_SFLOAT,
            TexFormat::Astc4x4Srgb => vk::Format::ASTC_4X4_SRGB_BLOCK,
            TexFormat::Astc4x4Unorm => vk::Format::ASTC_4X4_UNORM_BLOCK,
            TexFormat::Astc5x5Srgb => vk::Format::ASTC_5X5_SRGB_BLOCK,
            TexFormat::Astc5x5Unorm => vk::Format::ASTC_5X5_UNORM_BLOCK,
            TexFormat::Astc6x6Srgb => vk::Format::ASTC_6X6_SRGB_BLOCK,
            TexFormat::Astc6x6Unorm => vk::Format::ASTC_6X6_UNORM_BLOCK,
            TexFormat::Astc8x8Srgb => vk::Format::ASTC_8X8_SRGB_BLOCK,
            TexFormat::Astc8x8Unorm => vk::Format::ASTC_8X8_UNORM_BLOCK,
            TexFormat::Astc4x4Hdr => vk::Format::ASTC_4X4_SFLOAT_BLOCK,
            TexFormat::Bc1Srgb => vk::Format::BC1_RGB_SRGB_BLOCK,
            TexFormat::Bc1Unorm => vk::Format::BC1_RGB_UNORM_BLOCK,
            TexFormat::Bc4Unorm => vk::Format::BC4_UNORM_BLOCK,
            TexFormat::Bc5Unorm => vk::Format::BC5_UNORM_BLOCK,
        };
        // A block format the device can't sample would fail image creation
        // (or worse); the import cache never hands one over, so this is a
        // last line of defence against stale / foreign data.
        if (tex.format.is_bc() && !crate::bc_supported())
            || (tex.format.is_astc() && !crate::astc_supported())
            || (tex.format == TexFormat::Astc4x4Hdr && !crate::astc_hdr_supported())
        {
            anyhow::bail!("{:?} texture is not supported by this device", tex.format);
        }
        // Every mip must hold exactly its level's bytes: the copies below read
        // that many from the staging buffer.
        for l in layers {
            for (i, mip) in l.mips.iter().enumerate() {
                let (w, h) = ((tex.width >> i).max(1), (tex.height >> i).max(1));
                let want = tex.format.level_bytes(w, h);
                anyhow::ensure!(mip.len() == want, "{:?} mip {i} is {} bytes, expected {want}", tex.format, mip.len());
            }
        }
        let mip_levels = tex.mips.len().max(1) as u32;
        let extent = vk::Extent3D {
            width: tex.width.max(1),
            height: tex.height.max(1),
            depth: 1,
        };

        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(mip_levels)
            .array_layers(layer_count)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&info, None)? };
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "texture (bc)",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .with_context(|| "allocating GPU memory for texture (bc)")?;
        unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };

        // Pack all mips contiguously into one staging buffer.
        let total: usize = layers.iter().map(|l| l.mips.iter().map(|m| m.len()).sum::<usize>()).sum();
        let mut staging = Buffer::new(
            device,
            allocator,
            total.max(1) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "texture bc staging",
        )?;
        let mut offsets = Vec::with_capacity(layers.len() * tex.mips.len());
        let mut cursor = 0usize;
        for l in layers {
            for mip in &l.mips {
                staging.write(cursor, mip);
                offsets.push(cursor as u64);
                cursor += mip.len();
            }
        }

        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(mip_levels)
            .layer_count(layer_count);

        one_time_submit(device, pool, queue, |cb| unsafe {
            let to_transfer = vk::ImageMemoryBarrier2::default()
                .image(image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .subresource_range(range);
            device.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_transfer]),
            );

            let copies: Vec<vk::BufferImageCopy> = (0..layer_count * mip_levels)
                .map(|n| {
                    let (layer, level) = (n / mip_levels, n % mip_levels);
                    let w = (extent.width >> level).max(1);
                    let h = (extent.height >> level).max(1);
                    vk::BufferImageCopy::default()
                        .buffer_offset(offsets[n as usize])
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .mip_level(level)
                                .base_array_layer(layer)
                                .layer_count(1),
                        )
                        .image_extent(vk::Extent3D {
                            width: w,
                            height: h,
                            depth: 1,
                        })
                })
                .collect();
            device.cmd_copy_buffer_to_image(
                cb,
                staging.handle,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copies,
            );

            let to_sampled = vk::ImageMemoryBarrier2::default()
                .image(image)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .subresource_range(range);
            device.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_sampled]),
            );
        })?;
        staging.destroy(device, allocator);

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(if array || layer_count > 1 { vk::ImageViewType::TYPE_2D_ARRAY } else { vk::ImageViewType::TYPE_2D })
            .format(format)
            .subresource_range(range);
        let view = unsafe { device.create_image_view(&view_info, None)? };

        Ok(Self {
            image,
            view,
            allocation: Some(allocation),
        })
    }

    /// Upload a single-channel `R32_SFLOAT` 3D volume (one float per voxel,
    /// x fastest then y then z), used for per-mesh signed distance fields in
    /// the software-GI march. Sampled as a `sampler3D`. (Consumed once the SDF
    /// march pipeline lands; allow dead_code until then.)
    #[allow(dead_code)]
    pub fn upload_volume(
        device: &ash::Device,
        allocator: &mut Allocator,
        pool: vk::CommandPool,
        queue: vk::Queue,
        voxels: &[f32],
        dims: [u32; 3],
    ) -> Result<Self> {
        let format = vk::Format::R32_SFLOAT;
        let extent = vk::Extent3D {
            width: dims[0].max(1),
            height: dims[1].max(1),
            depth: dims[2].max(1),
        };
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_3D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&info, None)? };
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "mesh sdf",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .with_context(|| "allocating GPU memory for mesh sdf")?;
        unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };

        let mut staging = Buffer::new(
            device,
            allocator,
            std::mem::size_of_val(voxels) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "sdf staging",
        )?;
        staging.write(0, bytemuck::cast_slice(voxels));

        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        one_time_submit(device, pool, queue, |cb| unsafe {
            let to_transfer = vk::ImageMemoryBarrier2::default()
                .image(image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .subresource_range(range);
            device.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_transfer]),
            );
            let copy = vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_extent(extent);
            device.cmd_copy_buffer_to_image(
                cb,
                staging.handle,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy],
            );
            let to_sampled = vk::ImageMemoryBarrier2::default()
                .image(image)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .subresource_range(range);
            device.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_sampled]),
            );
        })?;
        staging.destroy(device, allocator);

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_3D)
            .format(format)
            .subresource_range(range);
        let view = unsafe { device.create_image_view(&view_info, None)? };
        Ok(Self {
            image,
            view,
            allocation: Some(allocation),
        })
    }

    /// Upload a `R32_UINT` 3D texture (the GDF nearest-instance index field).
    /// Same path as `upload_volume` but an unsigned-integer format, sampled with
    /// a NEAREST sampler as `usampler3D`.
    pub fn upload_volume_u32(
        device: &ash::Device,
        allocator: &mut Allocator,
        pool: vk::CommandPool,
        queue: vk::Queue,
        voxels: &[u32],
        dims: [u32; 3],
    ) -> Result<Self> {
        let format = vk::Format::R32_UINT;
        let extent = vk::Extent3D {
            width: dims[0].max(1),
            height: dims[1].max(1),
            depth: dims[2].max(1),
        };
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_3D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&info, None)? };
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "gdf index",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .with_context(|| "allocating GPU memory for gdf index")?;
        unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };

        let mut staging = Buffer::new(
            device,
            allocator,
            std::mem::size_of_val(voxels) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "gdf index staging",
        )?;
        staging.write(0, bytemuck::cast_slice(voxels));

        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        one_time_submit(device, pool, queue, |cb| unsafe {
            let to_transfer = vk::ImageMemoryBarrier2::default()
                .image(image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .subresource_range(range);
            device.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_transfer]),
            );
            let copy = vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_extent(extent);
            device.cmd_copy_buffer_to_image(
                cb,
                staging.handle,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy],
            );
            let to_sampled = vk::ImageMemoryBarrier2::default()
                .image(image)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .subresource_range(range);
            device.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_sampled]),
            );
        })?;
        staging.destroy(device, allocator);

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_3D)
            .format(format)
            .subresource_range(range);
        let view = unsafe { device.create_image_view(&view_info, None)? };
        Ok(Self {
            image,
            view,
            allocation: Some(allocation),
        })
    }

    /// Upload a baked lightmap array as a sampled 2D array in its own format
    /// (BC6H, ASTC HDR, E5B9G9R9, RGBA16F or RGBA32F), one layer per object. Short or
    /// long layers are padded / truncated to the exact layer size. Sampled in
    /// the standard shader by `uv1` + per-object layer for static GI.
    pub fn upload_lightmap_array(
        device: &ash::Device,
        allocator: &mut Allocator,
        pool: vk::CommandPool,
        queue: vk::Queue,
        array: &HdrTextureArray,
    ) -> Result<Self> {
        if !crate::HdrFormatSupport::current().can_sample(array.format, crate::astc_hdr_supported()) {
            anyhow::bail!("{} lightmaps are not supported by this device", array.format.label());
        }
        let format = array.format.vk_format();
        let size = array.size.max(1);
        let n = array.layers.len().max(1) as u32;
        let extent = vk::Extent3D {
            width: size,
            height: size,
            depth: 1,
        };
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(n)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&info, None)? };
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "lightmap array",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .with_context(|| "allocating GPU memory for lightmap array")?;
        unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };

        let layer_bytes = array.format.level_bytes(size, size);
        let mut staging = Buffer::new(
            device,
            allocator,
            (n as usize * layer_bytes) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "lightmap staging",
        )?;
        for (i, layer) in array.layers.iter().enumerate() {
            if layer.len() == layer_bytes {
                staging.write(i * layer_bytes, layer);
            } else {
                let mut buf = vec![0u8; layer_bytes];
                let k = layer.len().min(layer_bytes);
                buf[..k].copy_from_slice(&layer[..k]);
                staging.write(i * layer_bytes, &buf);
            }
        }

        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(n);

        one_time_submit(device, pool, queue, |cb| unsafe {
            let to_transfer = vk::ImageMemoryBarrier2::default()
                .image(image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .subresource_range(range);
            device.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_transfer]),
            );
            for i in 0..n {
                let copy = vk::BufferImageCopy::default()
                    .buffer_offset((i as usize * layer_bytes) as u64)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_array_layer(i)
                            .layer_count(1),
                    )
                    .image_extent(extent);
                device.cmd_copy_buffer_to_image(
                    cb,
                    staging.handle,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[copy],
                );
            }
            let to_sampled = vk::ImageMemoryBarrier2::default()
                .image(image)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .subresource_range(range);
            device.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_sampled]),
            );
        })?;
        staging.destroy(device, allocator);

        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
                    .format(format)
                    .subresource_range(range),
                None,
            )?
        };
        Ok(Self {
            image,
            view,
            allocation: Some(allocation),
        })
    }

    /// Move the GPU objects out into a new `GpuTexture`, leaving a null husk
    /// (deferred frees own the real one).
    pub fn take(&mut self) -> GpuTexture {
        GpuTexture {
            image: std::mem::take(&mut self.image),
            view: std::mem::take(&mut self.view),
            allocation: self.allocation.take(),
        }
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }
        if let Some(allocation) = self.allocation.take() {
            let _ = allocator.free(allocation);
        }
    }
}

/// Fill mips 1.. of `image` (`size` square, `levels` mips) from mip 0 by
/// linear blits, in `cb`. Mip 0 starts in `layout` after a write at
/// `src_stage` / `src_access`, the other mips in SHADER_READ_ONLY_OPTIMAL
/// (read at `readers`); all of them end in SHADER_READ_ONLY_OPTIMAL for
/// `readers`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn record_mip_chain(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
    size: u32,
    levels: u32,
    layout: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    readers: vk::PipelineStageFlags2,
) {
    let range = |base: u32, count: u32| {
        vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).base_mip_level(base).level_count(count).layer_count(1)
    };
    let barrier = |r: vk::ImageSubresourceRange, old: vk::ImageLayout, new: vk::ImageLayout, ss, sa, ds, da| {
        let b = [vk::ImageMemoryBarrier2::default()
            .image(image)
            .old_layout(old)
            .new_layout(new)
            .src_stage_mask(ss)
            .src_access_mask(sa)
            .dst_stage_mask(ds)
            .dst_access_mask(da)
            .subresource_range(r)];
        unsafe { device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&b)) };
    };
    let transfer = vk::PipelineStageFlags2::TRANSFER;
    let sampled = vk::AccessFlags2::SHADER_SAMPLED_READ;
    if levels <= 1 {
        barrier(range(0, 1), layout, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, src_stage, src_access, readers, sampled);
        return;
    }
    barrier(range(0, 1), layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, src_stage, src_access, transfer, vk::AccessFlags2::TRANSFER_READ);
    for level in 1..levels {
        barrier(
            range(level, 1),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            readers,
            vk::AccessFlags2::empty(),
            transfer,
            vk::AccessFlags2::TRANSFER_WRITE,
        );
        let edge = |l: u32| (size >> l).max(1) as i32;
        let layers = |l: u32| vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).mip_level(l).layer_count(1);
        let blit = vk::ImageBlit::default()
            .src_subresource(layers(level - 1))
            .src_offsets([vk::Offset3D::default(), vk::Offset3D { x: edge(level - 1), y: edge(level - 1), z: 1 }])
            .dst_subresource(layers(level))
            .dst_offsets([vk::Offset3D::default(), vk::Offset3D { x: edge(level), y: edge(level), z: 1 }]);
        unsafe {
            device.cmd_blit_image(
                cb,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::LINEAR,
            )
        };
        barrier(
            range(level, 1),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            transfer,
            vk::AccessFlags2::TRANSFER_WRITE,
            transfer,
            vk::AccessFlags2::TRANSFER_READ,
        );
    }
    barrier(
        range(0, levels),
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        transfer,
        vk::AccessFlags2::TRANSFER_READ,
        readers,
        sampled,
    );
}
