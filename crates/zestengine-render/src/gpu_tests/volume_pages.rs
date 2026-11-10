//! Persistent atlas copies, queue ordering and backpressure against a GPU
//! deliberately held behind a timeline semaphore. No renderer/pipeline warmup.
use ash::vk;
use gpu_allocator::MemoryLocation;
use crate::{alloc::{Buffer, one_time_submit}, texture::{GpuTexture, VolumePage, VolumePages}};

fn read(g: &crate::cluster_headless::HeadlessGpu, texture: &GpuTexture, buffer: &Buffer, dims: [u32; 3]) -> Vec<f32> {
    let device = &g.device;
    one_time_submit(device, g.pool, g.queue, |cb| unsafe {
        let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
        let begin = vk::ImageMemoryBarrier2::default().image(texture.image).subresource_range(range)
            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER).dst_access_mask(vk::AccessFlags2::TRANSFER_READ);
        device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&[begin]));
        let copy = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
            .image_extent(vk::Extent3D { width: dims[0], height: dims[1], depth: dims[2] });
        device.cmd_copy_image_to_buffer(cb, texture.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buffer.handle, &[copy]);
        let end = vk::ImageMemoryBarrier2::default().image(texture.image).subresource_range(range)
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL).new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER).src_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS).dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ);
        let host = vk::MemoryBarrier2::default().src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE).dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ);
        device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&[end]).memory_barriers(&[host]));
    }).unwrap();
    bytemuck::pod_collect_to_vec(&buffer.read()[..dims.iter().map(|&n| n as usize).product::<usize>() * 4])
}

#[test]
fn streamed_volume_pages_preserve_neighbors_and_apply_backpressure_without_waiting() {
    let Some(g) = super::gpu() else { return };
    let device = &g.device;
    assert!(g.timeline_semaphore, "timeline semaphore required for the asynchronous upload test");
    let before = g.validation_error_count();
    let mut alloc = g.allocator().unwrap();
    // Same deterministic family selection as HeadlessGpu's single queue.
    let family = unsafe { g.instance.get_physical_device_queue_family_properties(g.pd) }.iter()
        .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER)).unwrap() as u32;
    let dims = [16, 8, 8];
    let mut atlas = VolumePages::new(device, &mut alloc, g.queue, family, dims, 4096).unwrap();
    assert!(atlas.texture().is_none());
    assert_eq!(atlas.staging_bytes(), 0);
    let initial: Vec<_> = (0..1024).map(|i| i as f32 * 0.125 - 31.).collect();
    assert!(atlas.upload(device, &mut alloc, &[VolumePage { origin: [0; 3], size: dims, data: &initial }]).unwrap());
    let image = atlas.texture().unwrap().image;
    let mut buffer = Buffer::new(device, &mut alloc, 4096, vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuToCpu, "volume page validation").unwrap();
    assert_eq!(read(&g, atlas.texture().unwrap(), &buffer, dims), initial);
    atlas.collect(device, &mut alloc).unwrap();
    assert_eq!(atlas.staging_bytes(), 0);

    let semaphore = unsafe {
        let mut kind = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
        device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut kind), None).unwrap()
    };
    unsafe {
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&[1]);
        let lock = crate::alloc::queue_lock(g.queue);
        let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        device.queue_submit(g.queue, &[vk::SubmitInfo::default().wait_semaphores(&[semaphore])
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::ALL_COMMANDS]).push_next(&mut timeline)], vk::Fence::null()).unwrap();
    }
    // Release on timeout too, so a regression to blocking uploads fails rather
    // than permanently hanging the test process with a GPU waiting on the host.
    let (release, ready) = std::sync::mpsc::channel();
    let d = device.clone();
    let gate = std::thread::spawn(move || {
        let released = ready.recv_timeout(std::time::Duration::from_secs(5)).is_ok();
        unsafe { d.signal_semaphore(&vk::SemaphoreSignalInfo::default().semaphore(semaphore).value(1)).unwrap(); }
        released
    });
    let left = vec![7.25; 512];
    let right: Vec<_> = (0..512).map(|i| -0.5 - i as f32).collect();
    let a = atlas.upload(device, &mut alloc, &[VolumePage { origin: [0; 3], size: [8; 3], data: &left }]);
    let b = atlas.upload(device, &mut alloc, &[VolumePage { origin: [8, 0, 0], size: [8; 3], data: &right }]);
    let retained = atlas.staging_bytes();
    let retry = VolumePage { origin: [3, 2, 1], size: [1; 3], data: &[99.] };
    let c = atlas.upload(device, &mut alloc, std::slice::from_ref(&retry));
    let _ = release.send(());
    let returned_before_release = gate.join().unwrap();
    let actual = read(&g, atlas.texture().unwrap(), &buffer, dims);
    let mut expected = vec![0.; 1024];
    for z in 0..8 { for y in 0..8 { for x in 0..16 {
        expected[x + 16 * (y + 8 * z)] = if x < 8 { left[x + 8 * (y + 8 * z)] } else { right[x - 8 + 8 * (y + 8 * z)] };
    } } }
    assert!(returned_before_release, "upload blocked on GPU completion");
    assert!(a.unwrap() && b.unwrap());
    assert!(!c.unwrap(), "budget pressure must defer the whole request");
    assert_eq!(retained, 4096);
    assert_eq!(actual, expected, "deferred write changed atlas or queued writes reordered");
    atlas.collect(device, &mut alloc).unwrap();
    assert_eq!(atlas.staging_bytes(), 0);
    assert!(atlas.upload(device, &mut alloc, &[retry]).unwrap());
    expected[3 + 16 * (2 + 8)] = 99.;
    assert_eq!(read(&g, atlas.texture().unwrap(), &buffer, dims), expected, "page update changed neighboring texels");
    assert_eq!(atlas.texture().unwrap().image, image, "streaming replaced the atlas image");
    atlas.destroy(device, &mut alloc).unwrap();
    unsafe { device.destroy_semaphore(semaphore, None); }
    buffer.destroy(device, &mut alloc);
    drop(alloc);
    assert_eq!(g.validation_error_count(), before, "volume streaming produced Vulkan errors");
    eprintln!("[Flux volume pages] 3072 exact texels; bounded staging, deferred retry, stable image, no CPU upload wait");
}
