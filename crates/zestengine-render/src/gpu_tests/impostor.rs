//! The impostor GPU path on a real device: the atlas upload (array layers hold
//! exactly the texels of their views), and the billboard pipeline for every
//! scene-pass variant. The render comparison against the real mesh runs in the
//! real frame loop (`zestengine-assets/tests/cluster_frame.rs`), where the
//! capture code lives.

use ash::vk;
use gpu_allocator::MemoryLocation;

use super::gpu;
use crate::alloc::{self, Buffer};
use crate::impostor::{ImpostorGpu, ImpostorUpload, atlas_to_layers, create_layered_image};
use crate::pipeline::PassFormats;

/// A deterministic atlas whose texel encodes (view, row, column, channel layer).
fn test_atlas(grid: u32, tile: u32, salt: u8) -> Vec<u8> {
    let size = (grid * tile) as usize;
    let mut a = vec![0u8; size * size * 4];
    for y in 0..size {
        for x in 0..size {
            let (cell, ty, tx) = ((y / tile as usize) * grid as usize + x / tile as usize, y % tile as usize, x % tile as usize);
            let at = (y * size + x) * 4;
            a[at..at + 4].copy_from_slice(&[(cell as u8).wrapping_mul(7).wrapping_add(salt), ty as u8 * 5, tx as u8 * 3, 255 - (cell as u8)]);
        }
    }
    a
}

#[test]
fn atlas_upload_puts_every_view_in_its_own_layer() {
    let Some(g) = gpu() else { return };
    let mut alloc = g.allocator().unwrap();
    let (grid, tile) = (4u32, 16u32);
    let (albedo, aux) = (test_atlas(grid, tile, 1), test_atlas(grid, tile, 99));
    let mut texels = atlas_to_layers(grid, tile, &albedo);
    texels.extend_from_slice(&atlas_to_layers(grid, tile, &aux));
    let layers = 2 * grid * grid;
    let (image, view, allocation) = create_layered_image(&g.device, &mut alloc, g.pool, g.queue, tile, layers, &texels).unwrap();
    // Read every layer back.
    let size = texels.len();
    let mut host = Buffer::new(&g.device, &mut alloc, size as u64, vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuToCpu, "impostor readback").unwrap();
    alloc::one_time_submit(&g.device, g.pool, g.queue, |cb| unsafe {
        crate::image_barrier_layers(
            &g.device, cb, image, vk::ImageAspectFlags::COLOR, 0, layers,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ,
        );
        let copy = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(layers))
            .image_extent(vk::Extent3D { width: tile, height: tile, depth: 1 });
        g.device.cmd_copy_image_to_buffer(cb, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, host.handle, &[copy]);
    })
    .unwrap();
    let back = host.mapped().unwrap()[..size].to_vec();
    assert_eq!(back, texels, "the array image does not hold the uploaded texels");
    // And the layout means what it says: layer `cell` texel (y, x) is atlas texel (j*tile + y, i*tile + x).
    let (cell, y, x) = (6usize, 3usize, 9usize);
    let (i, j) = (cell % grid as usize, cell / grid as usize);
    let at = ((cell * tile as usize + y) * tile as usize + x) * 4;
    let src = (((j * tile as usize + y) * (grid * tile) as usize) + i * tile as usize + x) * 4;
    assert_eq!(back[at..at + 4], albedo[src..src + 4]);
    let aux_at = at + (grid * grid * tile * tile * 4) as usize;
    assert_eq!(back[aux_at..aux_at + 4], aux[src..src + 4], "aux views follow the albedo views");
    host.destroy(&g.device, &mut alloc);
    unsafe {
        g.device.destroy_image_view(view, None);
        g.device.destroy_image(image, None);
    }
    let _ = alloc.free(allocation);
    eprintln!("impostor atlas: {layers} layers of {tile}x{tile} read back identical");
}

#[test]
fn impostor_pipelines_exist_for_every_scene_pass_variant() {
    let Some(g) = gpu() else { return };
    let mut alloc = g.allocator().unwrap();
    // A set-0 layout like the frame set's (a uniform buffer at binding 0).
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)];
    let frame_layout = unsafe { g.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).unwrap() };
    let mut gpu_imp = ImpostorGpu::new(&g.device, &mut alloc, frame_layout).unwrap();
    let formats = [vk::Format::R8G8B8A8_UNORM, vk::Format::B8G8R8A8_SRGB, vk::Format::R16G16B16A16_SFLOAT, vk::Format::A2B10G10R10_UNORM_PACK32];
    let mut made = 0;
    for &color in &formats {
        for samples in [1u8, 4] {
            for (gbuf, velocity) in [(false, false), (true, false), (false, true), (true, true)] {
                for multiview in [false, true] {
                    if multiview && !g.multiview {
                        continue;
                    }
                    let pass = PassFormats { color, depth: vk::Format::D32_SFLOAT, gbuf: vk::Format::A2B10G10R10_UNORM_PACK32, multiview, samples };
                    gpu_imp.ensure_pipeline(&g.device, pass, gbuf, velocity).unwrap();
                    made += 1;
                }
            }
        }
    }
    // An upload through the real path, and the limits.
    let (grid, tile) = (4u32, 16u32);
    let (albedo, aux) = (test_atlas(grid, tile, 3), test_atlas(grid, tile, 4));
    let up = ImpostorUpload { grid, tile, centre: [0.0, 1.0, 0.0], radius: 1.0, albedo: &albedo, aux: &aux };
    let h = gpu_imp.upload(&g.device, &mut alloc, g.pool, g.queue, &up).unwrap();
    assert_eq!(gpu_imp.image_of(h).unwrap().1, 2 * grid * grid);
    let too_many_views = ImpostorUpload { grid: 12, ..up };
    assert!(gpu_imp.upload(&g.device, &mut alloc, g.pool, g.queue, &too_many_views).is_err());
    gpu_imp.free(&g.device, &mut alloc, h);
    assert!(gpu_imp.image_of(h).is_none());
    eprintln!("impostor: {made} pipeline variants created (multiview device support: {})", g.multiview);
    gpu_imp.destroy(&g.device, &mut alloc);
    unsafe { g.device.destroy_descriptor_set_layout(frame_layout, None) };
}
