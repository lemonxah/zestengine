#![allow(dead_code)]

use ash::vk;

use crate::sync::{Access, Range, SyncBatch, Tracker, is_write};

pub struct Pass<'t> {
    tracker: &'t mut Tracker,
    name: &'static str,
    images: Vec<(vk::Image, Range, Access)>,
    buffers: Vec<(vk::Buffer, Access)>,
}

pub fn pass<'t>(tracker: &'t mut Tracker, name: &'static str) -> Pass<'t> {
    Pass { tracker, name, images: Vec::new(), buffers: Vec::new() }
}

fn overlaps(a: Range, b: Range) -> bool {
    match (a, b) {
        (Range::All, _) | (_, Range::All) => true,
        (Range::Layers { base: a0, count: ac }, Range::Layers { base: b0, count: bc }) | (Range::Mips { base: a0, count: ac }, Range::Mips { base: b0, count: bc }) => a0 < b0 + bc && b0 < a0 + ac,
        _ => true,
    }
}

impl<'t> Pass<'t> {
    pub fn image(self, image: vk::Image, access: Access) -> Self {
        self.image_range(image, Range::All, access)
    }

    pub fn image_range(mut self, image: vk::Image, range: Range, access: Access) -> Self {
        let clash = self
            .images
            .iter()
            .find(|(other, other_range, other_access)| *other == image && overlaps(*other_range, range) && *other_access != access && (is_write(*other_access) || is_write(access) || crate::sync::info(*other_access).layout != crate::sync::info(access).layout))
            .map(|(_, _, a)| *a);
        match clash {
            Some(first) => {
                let message = format!("pass {:?} declares one image as both {first:?} and {access:?}; a pass states what it needs now, not a sequence", self.name);
                debug_assert!(false, "{message}");
                tracing::error!(target: "sync", "{message}");
            }
            None => {
                if !self.images.contains(&(image, range, access)) {
                    self.images.push((image, range, access));
                }
            }
        }
        self
    }

    pub fn buffer(mut self, buffer: vk::Buffer, access: Access) -> Self {
        self.buffers.push((buffer, access));
        self
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn writes(&self) -> usize {
        self.images.iter().filter(|(_, _, access)| is_write(*access)).count()
    }

    fn apply(self) -> &'t mut Tracker {
        let Pass { tracker, images, buffers, .. } = self;
        for (image, range, access) in images {
            tracker.require(image, range, access);
        }
        for (buffer, access) in buffers {
            tracker.require_buffer(buffer, access);
        }
        tracker
    }

    pub fn declare(self) -> SyncBatch {
        self.apply().drain()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::{ImageDesc, Lifetime, Mode};
    use ash::vk::Handle;

    fn swapchain(tracker: &mut Tracker) -> vk::Image {
        let image = vk::Image::from_raw(0x5a);
        tracker.import_image(
            image,
            ImageDesc { name: "swapchain", aspect: vk::ImageAspectFlags::COLOR, mips: 1, layers: 1, lifetime: Lifetime::Imported },
            vk::ImageLayout::UNDEFINED,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        );
        image
    }

    fn one(batch: &SyncBatch) -> vk::ImageMemoryBarrier2<'static> {
        assert_eq!(batch.images.len(), 1, "one barrier expected");
        batch.images[0]
    }

    #[test]
    fn the_direct_swapchain_frame_is_acquire_then_colour_then_capture_then_present() {
        use vk::ImageLayout as L;
        use vk::PipelineStageFlags2 as S;
        let mut tracker = Tracker::with_mode(Mode::Emit);
        let image = swapchain(&mut tracker);
        let first = one(&pass(&mut tracker, "tonemap").image(image, Access::ColorWrite).declare());
        assert_eq!((first.old_layout, first.new_layout), (L::UNDEFINED, L::COLOR_ATTACHMENT_OPTIMAL));
        assert!(first.src_stage_mask.contains(S::COLOR_ATTACHMENT_OUTPUT), "the first use waits for the acquire semaphore's stage");
        let to_copy = one(&pass(&mut tracker, "capture").image(image, Access::TransferSrc).declare());
        assert_eq!((to_copy.old_layout, to_copy.new_layout), (L::COLOR_ATTACHMENT_OPTIMAL, L::TRANSFER_SRC_OPTIMAL));
        assert!(to_copy.src_stage_mask.contains(S::COLOR_ATTACHMENT_OUTPUT) && to_copy.src_access_mask.contains(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE));
        let back = one(&pass(&mut tracker, "after capture").image(image, Access::ColorReadWrite).declare());
        assert_eq!((back.old_layout, back.new_layout), (L::TRANSFER_SRC_OPTIMAL, L::COLOR_ATTACHMENT_OPTIMAL));
        assert!(back.src_stage_mask.contains(S::TRANSFER));
        let present = one(&pass(&mut tracker, "present").image(image, Access::Present).declare());
        assert_eq!((present.old_layout, present.new_layout), (L::COLOR_ATTACHMENT_OPTIMAL, L::PRESENT_SRC_KHR));
        assert!(present.src_stage_mask.contains(S::COLOR_ATTACHMENT_OUTPUT));
        assert_eq!(tracker.present_signal_stage(image), S::ALL_COMMANDS);
        assert!(pass(&mut tracker, "present again").image(image, Access::Present).declare().is_empty(), "nothing to do when the image is already there");
    }

    #[test]
    fn the_frame_generation_path_blits_then_draws_the_ui_over_it() {
        use vk::ImageLayout as L;
        let mut tracker = Tracker::with_mode(Mode::Emit);
        let image = swapchain(&mut tracker);
        let blit = one(&pass(&mut tracker, "blit").image(image, Access::TransferDst).declare());
        assert_eq!((blit.old_layout, blit.new_layout), (L::UNDEFINED, L::TRANSFER_DST_OPTIMAL));
        assert!(blit.src_stage_mask.contains(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT));
        let ui = one(&pass(&mut tracker, "ui").image(image, Access::ColorReadWrite).declare());
        assert_eq!((ui.old_layout, ui.new_layout), (L::TRANSFER_DST_OPTIMAL, L::COLOR_ATTACHMENT_OPTIMAL));
        assert!(ui.src_access_mask.contains(vk::AccessFlags2::TRANSFER_WRITE), "the UI waits for the blit's writes");
        assert!(ui.dst_access_mask.contains(vk::AccessFlags2::COLOR_ATTACHMENT_READ), "blending reads what the blit wrote");
    }

    #[test]
    fn a_new_acquire_starts_the_image_over_and_unknown_images_emit_nothing() {
        let mut tracker = Tracker::with_mode(Mode::Emit);
        let image = swapchain(&mut tracker);
        pass(&mut tracker, "tonemap").image(image, Access::ColorWrite).declare();
        pass(&mut tracker, "present").image(image, Access::Present).declare();
        let image = swapchain(&mut tracker);
        let again = one(&pass(&mut tracker, "tonemap").image(image, Access::ColorWrite).declare());
        assert_eq!(again.old_layout, vk::ImageLayout::UNDEFINED, "last frame's present layout is not assumed");
        let stranger = vk::Image::from_raw(0x99);
        assert!(pass(&mut tracker, "stray").image(stranger, Access::ColorWrite).declare().is_empty());
    }

    #[test]
    fn a_pass_batches_its_images_and_counts_its_writes() {
        let mut tracker = Tracker::with_mode(Mode::Emit);
        let desc = |name| ImageDesc { name, aspect: vk::ImageAspectFlags::COLOR, mips: 1, layers: 4, lifetime: Lifetime::Shared };
        let (hdr, gbuf, atlas) = (vk::Image::from_raw(1), vk::Image::from_raw(2), vk::Image::from_raw(3));
        tracker.register_image(hdr, desc("hdr"));
        tracker.register_image(gbuf, desc("gbuf"));
        tracker.register_image(atlas, desc("atlas"));
        let scene = pass(&mut tracker, "scene")
            .image(hdr, Access::ColorWrite)
            .image(gbuf, Access::ColorWrite)
            .image(hdr, Access::ColorWrite)
            .image_range(atlas, Range::Layers { base: 0, count: 2 }, Access::SampledFragment)
            .image_range(atlas, Range::Layers { base: 2, count: 2 }, Access::DepthWrite);
        assert_eq!((scene.name(), scene.writes()), ("scene", 3));
        let batch = scene.declare();
        assert_eq!(batch.submission_count(), 1, "one barrier call for the whole pass");
        assert!(batch.images.len() >= 4, "hdr, gbuf and both halves of the atlas: {}", batch.images.len());
        let resolve = pass(&mut tracker, "resolve").image(hdr, Access::SampledCompute).image(gbuf, Access::SampledCompute).declare();
        assert_eq!(resolve.images.len(), 2);
        assert!(resolve.images.iter().all(|b| b.new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL));
    }

    #[test]
    fn two_readers_of_one_image_share_a_pass() {
        let mut tracker = Tracker::with_mode(Mode::Emit);
        let hdr = vk::Image::from_raw(7);
        tracker.register_image(hdr, ImageDesc { name: "hdr", aspect: vk::ImageAspectFlags::COLOR, mips: 1, layers: 1, lifetime: Lifetime::Shared });
        pass(&mut tracker, "scene").image(hdr, Access::ColorWrite).declare();
        let batch = pass(&mut tracker, "post and resolve").image(hdr, Access::SampledFragment).image(hdr, Access::SampledCompute).declare();
        let stages = batch.images.iter().fold(vk::PipelineStageFlags2::empty(), |all, b| all | b.dst_stage_mask);
        assert!(stages.contains(vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER), "both readers are covered: 0x{:x}", stages.as_raw());
        assert!(batch.images.iter().all(|b| b.new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL));
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "declares one image as both"))]
    fn one_image_cannot_be_two_things_in_one_pass() {
        let mut tracker = Tracker::with_mode(Mode::Emit);
        let image = swapchain(&mut tracker);
        let batch = pass(&mut tracker, "confused").image(image, Access::ColorWrite).image(image, Access::SampledFragment).declare();
        assert_eq!(batch.images.len(), 1, "release builds keep the first declaration");
    }
}
