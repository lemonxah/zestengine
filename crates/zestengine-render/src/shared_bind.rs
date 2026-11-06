//! Swapping a resource that the set-0 descriptor sets point at (baked probes,
//! lightmap array, reflection cubes, shadow atlas, opaque copies) without a
//! `vkDeviceWaitIdle` (PERF-2).
//!
//! A frame slot's sets (frame set, camera-preview set, reflection-capture and
//! planar sets) are in use by the GPU until that slot's fence signals, and a
//! descriptor set may not be rewritten while a pending command buffer uses it.
//! So a swap writes the new binding at once into the sets of every slot whose
//! fence has already signalled, and queues the same writes for the others;
//! `Renderer::render_frame` applies a slot's queue right after it waits for the
//! slot's fence, before anything is recorded. The replaced resource is parked
//! with the serial of its swap and destroyed once its countdown (rendered
//! frames, see `defer_free`) has expired AND every queued write up to that serial
//! has been applied, so a skipped frame can never free a resource a set still
//! names.

use super::*;

/// Which sets of a slot a write goes to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SharedTarget {
    /// Frame set, preview set, capture sets and planar sets (`shared_binding_sets`).
    All,
    /// Only the frame set.
    FrameSet,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum SharedResource {
    Buffer(vk::Buffer),
    Image { sampler: vk::Sampler, view: vk::ImageView, layout: vk::ImageLayout },
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SharedWrite {
    pub target: SharedTarget,
    pub binding: u32,
    pub resource: SharedResource,
}

impl SharedWrite {
    /// A storage buffer (whole range).
    pub fn buffer(target: SharedTarget, binding: u32, buffer: vk::Buffer) -> Self {
        Self { target, binding, resource: SharedResource::Buffer(buffer) }
    }

    /// A sampled image (combined image sampler) in the shader-read layout.
    pub fn sampled(target: SharedTarget, binding: u32, sampler: vk::Sampler, view: vk::ImageView) -> Self {
        Self { target, binding, resource: SharedResource::Image { sampler, view, layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL } }
    }
}

/// Writes still owed to the slots in `slots` (bit per frame slot).
pub(crate) struct PendingShared {
    slots: u32,
    seq: u64,
    writes: Vec<SharedWrite>,
}

impl Renderer {
    /// The sets of one frame slot a write with `target` reaches.
    fn slot_sets(&self, slot: usize, target: SharedTarget) -> Vec<vk::DescriptorSet> {
        let mut sets = Vec::new();
        if matches!(target, SharedTarget::All | SharedTarget::FrameSet) {
            sets.extend(self.frame_sets.get(slot).copied());
        }
        if target == SharedTarget::All
            && let Some(preview) = &self.camera_preview
        {
            sets.extend(preview.sets.get(slot).copied());
        }
        if target == SharedTarget::All {
            sets.extend(self.refl_capture_sets.iter().skip(slot * 6).take(6).copied());
            sets.extend(self.planar_sets.iter().skip(slot * MAX_PLANAR).take(MAX_PLANAR).copied());
        }
        sets
    }

    fn write_slot(&self, slot: usize, writes: &[SharedWrite]) {
        for w in writes {
            for set in self.slot_sets(slot, w.target) {
                match w.resource {
                    SharedResource::Buffer(buffer) => {
                        let info = [vk::DescriptorBufferInfo::default().buffer(buffer).range(vk::WHOLE_SIZE)];
                        let write = vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(w.binding)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&info);
                        unsafe { self.ctx.device.update_descriptor_sets(&[write], &[]) };
                    }
                    SharedResource::Image { sampler, view, layout } => {
                        let info = [vk::DescriptorImageInfo::default().sampler(sampler).image_view(view).image_layout(layout)];
                        let write = vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(w.binding)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(&info);
                        unsafe { self.ctx.device.update_descriptor_sets(&[write], &[]) };
                    }
                }
            }
        }
    }

    /// Point `writes` at the new resources in every slot's sets: now where the
    /// slot's last submission has retired, queued for the others. Returns the
    /// swap's serial for [`Renderer::defer_shared_free`]. `in_frame`: called
    /// from inside `render_frame` after its fence wait, where the slot being
    /// recorded counts as retired (its command buffer has not bound anything yet).
    pub(crate) fn queue_shared_writes(&mut self, writes: Vec<SharedWrite>, in_frame: bool) -> u64 {
        self.shared_seq += 1;
        let seq = self.shared_seq;
        let mut owed = 0u32;
        for slot in 0..FRAMES_IN_FLIGHT {
            let retired = (in_frame && slot == self.frame_index)
                || self.frames.get(slot).is_some_and(|f| unsafe { self.ctx.device.get_fence_status(f.in_flight) }.unwrap_or(false));
            if retired {
                // Older swaps first, so the newest resource wins.
                self.apply_shared_slot(slot);
                self.write_slot(slot, &writes);
            } else {
                owed |= 1 << slot;
            }
        }
        if owed != 0 {
            self.pending_shared.push(PendingShared { slots: owed, seq, writes });
        }
        seq
    }

    /// Run the writes queued for `slot`. Called by `render_frame` right after the
    /// slot's fence wait (and by [`Renderer::apply_all_shared`] with the device idle).
    pub(crate) fn apply_shared_slot(&mut self, slot: usize) {
        if self.pending_shared.is_empty() {
            return;
        }
        let mut pending = std::mem::take(&mut self.pending_shared);
        for p in &mut pending {
            if p.slots & (1 << slot) != 0 {
                self.write_slot(slot, &p.writes);
                p.slots &= !(1 << slot);
            }
        }
        pending.retain(|p| p.slots != 0);
        self.pending_shared = pending;
    }

    /// Apply every queued write to every slot (the device is idle).
    pub(crate) fn apply_all_shared(&mut self) {
        for slot in 0..FRAMES_IN_FLIGHT {
            self.apply_shared_slot(slot);
        }
    }

    /// Park the destroy of a resource a queued swap replaced.
    pub(crate) fn defer_shared_free(&mut self, seq: u64, f: impl FnOnce(&ash::Device, &mut Allocator) + Send + 'static) {
        self.shared_frees.push((FRAMES_IN_FLIGHT + 1, seq, Box::new(f)));
    }

    /// Count the shared frees down one rendered frame and run the ones whose
    /// countdown expired and whose swap every slot has applied; `all` runs
    /// everything (the device is idle).
    pub(crate) fn run_shared_frees(&mut self, all: bool) {
        if self.shared_frees.is_empty() {
            return;
        }
        for (frames, ..) in &mut self.shared_frees {
            *frames = frames.saturating_sub(1);
        }
        let Some(allocator) = self.allocator.clone() else { return };
        let mut alloc = allocator.lock().unwrap();
        let mut i = 0;
        while i < self.shared_frees.len() {
            let (frames, seq, _) = &self.shared_frees[i];
            let applied = !self.pending_shared.iter().any(|p| p.seq <= *seq);
            if all || (*frames == 0 && applied) {
                let (_, _, f) = self.shared_frees.remove(i);
                f(&self.ctx.device, &mut alloc);
            } else {
                i += 1;
            }
        }
    }
}
