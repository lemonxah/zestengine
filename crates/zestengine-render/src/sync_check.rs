use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, Ordering};

use ash::vk;

use crate::sync::{Access, ImageDesc, Lifetime, Mismatch, Mode, Range, Tracker, mode_from_env};

const UNSET: u8 = 0;
const OFF: u8 = 1;
const ON: u8 = 2;

static CHECKING: AtomicU8 = AtomicU8::new(UNSET);
static TRACKER: Mutex<Option<Tracker>> = Mutex::new(None);

const WRITES: vk::AccessFlags2 = vk::AccessFlags2::from_raw(
    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw()
        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw()
        | vk::AccessFlags2::SHADER_WRITE.as_raw()
        | vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw()
        | vk::AccessFlags2::TRANSFER_WRITE.as_raw()
        | vk::AccessFlags2::HOST_WRITE.as_raw()
        | vk::AccessFlags2::MEMORY_WRITE.as_raw()
        | vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR.as_raw(),
);

fn new_tracker(panic: bool) -> Tracker {
    if panic { Tracker::panicking() } else { Tracker::with_mode(Mode::Emit) }
}

pub fn checking() -> bool {
    match CHECKING.load(Ordering::Relaxed) {
        ON => true,
        OFF => false,
        _ => init(),
    }
}

#[cold]
fn init() -> bool {
    let (mode, panic) = mode_from_env();
    set_mode(mode, panic)
}

pub fn set_mode(mode: Mode, panic: bool) -> bool {
    let on = mode == Mode::Check;
    let mut tracker = TRACKER.lock().unwrap_or_else(|e| e.into_inner());
    if tracker.is_none() || panic {
        *tracker = Some(new_tracker(on && panic));
    }
    CHECKING.store(if on { ON } else { OFF }, Ordering::Relaxed);
    on
}

fn with<R>(run: impl FnOnce(&mut Tracker) -> R) -> R {
    let mut guard = TRACKER.lock().unwrap_or_else(|e| e.into_inner());
    run(guard.get_or_insert_with(|| new_tracker(false)))
}

pub fn barrier_access(barrier: &vk::ImageMemoryBarrier2<'_>) -> Access {
    if barrier.new_layout == vk::ImageLayout::PRESENT_SRC_KHR {
        Access::Present
    } else if barrier.dst_access_mask.intersects(WRITES) {
        Access::ColorWrite
    } else {
        Access::SampledFragment
    }
}

pub fn shared(name: &'static str, aspect: vk::ImageAspectFlags, mips: u32, layers: u32) -> ImageDesc {
    ImageDesc { name, aspect, mips: mips.max(1), layers: layers.max(1), lifetime: Lifetime::Shared }
}

pub fn register(image: vk::Image, desc: ImageDesc) {
    if image != vk::Image::null() {
        with(|tracker| tracker.register_image(image, desc));
    }
}

pub fn import(image: vk::Image, desc: ImageDesc, layout: vk::ImageLayout, after: vk::PipelineStageFlags2) {
    if image != vk::Image::null() {
        with(|tracker| tracker.import_image(image, desc, layout, after));
    }
}

pub fn forget(image: vk::Image) {
    if image != vk::Image::null() {
        with(|tracker| tracker.forget_image(image));
    }
}

pub fn tracked(image: vk::Image) -> bool {
    with(|tracker| tracker.image_name(image).is_some())
}

pub fn present_signal_stage(image: vk::Image) -> vk::PipelineStageFlags2 {
    with(|tracker| tracker.present_signal_stage(image))
}

pub fn pass(device: &ash::Device, cb: vk::CommandBuffer, name: &'static str, uses: &[(vk::Image, Access)]) {
    pass_ranges(device, cb, name, uses.iter().map(|(image, access)| (*image, Range::All, *access)));
}

pub fn pass_ranges(device: &ash::Device, cb: vk::CommandBuffer, name: &'static str, uses: impl IntoIterator<Item = (vk::Image, Range, Access)>) {
    let batch = with(|tracker| {
        let mut pass = crate::graph::pass(tracker, name);
        for (image, range, access) in uses {
            pass = pass.image_range(image, range, access);
        }
        pass.declare()
    });
    if batch.is_empty() {
        return;
    }
    let memory = batch.memory.map(|barrier| [barrier]);
    let memory = memory.as_ref().map_or(&[][..], |barrier| barrier.as_slice());
    let dependency = vk::DependencyInfo::default().image_memory_barriers(&batch.images).memory_barriers(memory);
    unsafe { device.cmd_pipeline_barrier2(cb, &dependency) };
}

type Site = &'static std::panic::Location<'static>;

fn env_flag(cell: &std::sync::OnceLock<bool>, name: &str) -> bool {
    *cell.get_or_init(|| std::env::var_os(name).is_some())
}

fn observe_at(barriers: &[vk::ImageMemoryBarrier2<'_>], site: Site) {
    static SITES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if barriers.is_empty() {
        return;
    }
    let check = checking();
    let list_sites = check && env_flag(&SITES, "ZESTENGINE_SYNC_CHECK_SITES");
    let (found, used) = with(|tracker| {
        let mut used = Vec::new();
        for barrier in barriers {
            if list_sites && let Some(name) = tracker.image_name(barrier.image) {
                used.push((name, barrier.new_layout));
            }
            tracker.follow(barrier, barrier_access(barrier), check);
        }
        (if check { tracker.take_mismatches() } else { Vec::new() }, used)
    });
    if !used.is_empty() {
        report_sites(&used, site);
    }
    if !found.is_empty() {
        report(found, site);
    }
}

fn report_sites(used: &[(&'static str, vk::ImageLayout)], site: Site) {
    static SEEN: Mutex<Option<std::collections::HashSet<(&'static str, i32, &'static str, u32)>>> = Mutex::new(None);
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let seen = seen.get_or_insert_with(Default::default);
    for (name, layout) in used {
        if seen.insert((name, layout.as_raw(), site.file(), site.line())) {
            eprintln!("sync site: {name} -> {layout:?} at {}:{}", site.file(), site.line());
        }
    }
}

fn report(found: Vec<Mismatch>, site: Site) {
    static PRINT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static SEEN: Mutex<Option<std::collections::HashSet<(String, String)>>> = Mutex::new(None);
    if !env_flag(&PRINT, "ZESTENGINE_SYNC_CHECK_PRINT") {
        return;
    }
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let seen = seen.get_or_insert_with(Default::default);
    for mismatch in found {
        let kind = format!("{:?}", mismatch.kind);
        if seen.insert((mismatch.name.to_owned(), format!("{kind} {} {}:{}", mismatch.detail, site.file(), site.line()))) {
            eprintln!("sync check: {}: {kind}: {} at {}:{}", mismatch.name, mismatch.detail, site.file(), site.line());
        }
    }
}

#[track_caller]
pub unsafe fn barrier2(device: &ash::Device, cb: vk::CommandBuffer, dependency: &vk::DependencyInfo<'_>) {
    if dependency.image_memory_barrier_count > 0 && !dependency.p_image_memory_barriers.is_null() {
        let barriers = unsafe { std::slice::from_raw_parts(dependency.p_image_memory_barriers, dependency.image_memory_barrier_count as usize) };
        observe_at(barriers, std::panic::Location::caller());
    }
    unsafe { device.cmd_pipeline_barrier2(cb, dependency) };
}

#[allow(clippy::too_many_arguments)]
#[track_caller]
pub unsafe fn barrier1(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    src: vk::PipelineStageFlags,
    dst: vk::PipelineStageFlags,
    flags: vk::DependencyFlags,
    memory: &[vk::MemoryBarrier<'_>],
    buffers: &[vk::BufferMemoryBarrier<'_>],
    images: &[vk::ImageMemoryBarrier<'_>],
) {
    if !images.is_empty() {
        let converted: Vec<vk::ImageMemoryBarrier2<'_>> = images
            .iter()
            .map(|b| {
                vk::ImageMemoryBarrier2::default()
                    .image(b.image)
                    .old_layout(b.old_layout)
                    .new_layout(b.new_layout)
                    .src_stage_mask(vk::PipelineStageFlags2::from_raw(src.as_raw() as u64))
                    .src_access_mask(vk::AccessFlags2::from_raw(b.src_access_mask.as_raw() as u64))
                    .dst_stage_mask(vk::PipelineStageFlags2::from_raw(dst.as_raw() as u64))
                    .dst_access_mask(vk::AccessFlags2::from_raw(b.dst_access_mask.as_raw() as u64))
                    .subresource_range(b.subresource_range)
            })
            .collect();
        observe_at(&converted, std::panic::Location::caller());
    }
    unsafe { device.cmd_pipeline_barrier(cb, src, dst, flags, memory, buffers, images) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    fn barrier(image: vk::Image, old: vk::ImageLayout, new: vk::ImageLayout, src: vk::PipelineStageFlags2, dst: vk::PipelineStageFlags2, access: vk::AccessFlags2) -> vk::ImageMemoryBarrier2<'static> {
        vk::ImageMemoryBarrier2::default()
            .image(image)
            .old_layout(old)
            .new_layout(new)
            .src_stage_mask(src)
            .dst_stage_mask(dst)
            .dst_access_mask(access)
            .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1))
    }

    #[test]
    fn destination_access_is_classified() {
        use vk::AccessFlags2 as A;
        use vk::ImageLayout as L;
        use vk::PipelineStageFlags2 as S;
        let image = vk::Image::from_raw(0x51);
        let write = barrier(image, L::UNDEFINED, L::COLOR_ATTACHMENT_OPTIMAL, S::TOP_OF_PIPE, S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE);
        assert_eq!(barrier_access(&write), Access::ColorWrite);
        let read = barrier(image, L::COLOR_ATTACHMENT_OPTIMAL, L::SHADER_READ_ONLY_OPTIMAL, S::COLOR_ATTACHMENT_OUTPUT, S::COMPUTE_SHADER, A::SHADER_SAMPLED_READ);
        assert_eq!(barrier_access(&read), Access::SampledFragment);
        let storage = barrier(image, L::GENERAL, L::GENERAL, S::COMPUTE_SHADER, S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE | A::SHADER_STORAGE_READ);
        assert_eq!(barrier_access(&storage), Access::ColorWrite);
        let present = barrier(image, L::COLOR_ATTACHMENT_OPTIMAL, L::PRESENT_SRC_KHR, S::COLOR_ATTACHMENT_OUTPUT, S::empty(), A::empty());
        assert_eq!(barrier_access(&present), Access::Present);
    }

    #[test]
    fn a_shared_target_reused_without_waiting_for_its_readers_is_flagged() {
        use vk::AccessFlags2 as A;
        use vk::ImageLayout as L;
        use vk::PipelineStageFlags2 as S;
        let image = vk::Image::from_raw(0x77);
        let mut tracker = Tracker::with_mode(Mode::Check);
        tracker.register_image(image, shared("viewport hdr", vk::ImageAspectFlags::COLOR, 1, 1));
        let frame = [
            barrier(image, L::UNDEFINED, L::COLOR_ATTACHMENT_OPTIMAL, S::TOP_OF_PIPE, S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE),
            barrier(image, L::COLOR_ATTACHMENT_OPTIMAL, L::SHADER_READ_ONLY_OPTIMAL, S::COLOR_ATTACHMENT_OUTPUT, S::COMPUTE_SHADER, A::SHADER_SAMPLED_READ),
        ];
        for b in &frame {
            tracker.follow(b, barrier_access(b), true);
        }
        assert!(tracker.mismatches().is_empty(), "the first frame has no prior use: {:?}", tracker.mismatches());
        tracker.follow(&frame[0], barrier_access(&frame[0]), true);
        assert!(!tracker.mismatches().is_empty(), "reusing the target from TOP_OF_PIPE skips last frame's compute reader");
        let mut tracker = Tracker::with_mode(Mode::Check);
        tracker.register_image(image, shared("viewport hdr", vk::ImageAspectFlags::COLOR, 1, 1));
        for b in &frame {
            tracker.follow(b, barrier_access(b), true);
        }
        let waits = barrier(image, L::UNDEFINED, L::COLOR_ATTACHMENT_OPTIMAL, S::COLOR_ATTACHMENT_OUTPUT | S::COMPUTE_SHADER, S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE);
        tracker.follow(&waits, barrier_access(&waits), true);
        assert!(tracker.mismatches().is_empty(), "{:?}", tracker.mismatches());
    }
}
