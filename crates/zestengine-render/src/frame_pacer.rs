use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::time::Duration;

use ash::{khr, vk};

use crate::frame::FRAMES_IN_FLIGHT;

pub(crate) struct PacedFrame {
    pub slot: usize,
    pub fence: vk::Fence,
    pub loader: khr::swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub queue: vk::Queue,
    pub generated: Option<(u32, vk::Semaphore)>,
    pub real: (u32, vk::Semaphore),
    pub gap: Duration,
}

struct Shared {
    busy: Mutex<[u32; FRAMES_IN_FLIGHT]>,
    idle: Condvar,
    resize: AtomicBool,
    real: AtomicU64,
    generated: AtomicU64,
}

pub(crate) struct FramePacer {
    jobs: Option<mpsc::Sender<PacedFrame>>,
    shared: Arc<Shared>,
    thread: Option<std::thread::JoinHandle<()>>,
}

fn present(frame: &PacedFrame, image: u32, wait: vk::Semaphore, shared: &Shared) {
    let waits = [wait];
    let swapchains = [frame.swapchain];
    let indices = [image];
    let info = vk::PresentInfoKHR::default().wait_semaphores(&waits).swapchains(&swapchains).image_indices(&indices);
    let lock = crate::alloc::queue_lock(frame.queue);
    let result = {
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { frame.loader.queue_present(frame.queue, &info) }
    };
    match result {
        Ok(false) => {}
        Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => shared.resize.store(true, Ordering::Relaxed),
        Err(e) => tracing::warn!("frame generation present failed: {e}"),
    }
}

fn run(device: ash::Device, jobs: mpsc::Receiver<PacedFrame>, shared: Arc<Shared>) {
    for frame in jobs {
        if let Err(e) = unsafe { device.wait_for_fences(&[frame.fence], true, 2_000_000_000) } {
            tracing::warn!("frame generation: waiting for the frame failed: {e}");
        }
        if let Some((image, wait)) = frame.generated {
            present(&frame, image, wait, &shared);
            shared.generated.fetch_add(1, Ordering::Relaxed);
            std::thread::sleep(frame.gap);
        }
        present(&frame, frame.real.0, frame.real.1, &shared);
        shared.real.fetch_add(1, Ordering::Relaxed);
        let mut busy = shared.busy.lock().unwrap_or_else(|e| e.into_inner());
        busy[frame.slot] = busy[frame.slot].saturating_sub(1);
        shared.idle.notify_all();
    }
}

impl FramePacer {
    pub fn new(device: ash::Device) -> std::io::Result<Self> {
        let shared = Arc::new(Shared {
            busy: Mutex::new([0; FRAMES_IN_FLIGHT]),
            idle: Condvar::new(),
            resize: AtomicBool::new(false),
            real: AtomicU64::new(0),
            generated: AtomicU64::new(0),
        });
        let (jobs, received) = mpsc::channel();
        let thread_shared = shared.clone();
        let thread = std::thread::Builder::new().name("frame pacer".into()).spawn(move || run(device, received, thread_shared))?;
        Ok(Self { jobs: Some(jobs), shared, thread: Some(thread) })
    }

    pub fn submit(&self, frame: PacedFrame) -> Result<(), PacedFrame> {
        let slot = frame.slot;
        self.shared.busy.lock().unwrap_or_else(|e| e.into_inner())[slot] += 1;
        let Some(jobs) = self.jobs.as_ref() else { return Err(frame) };
        jobs.send(frame).map_err(|returned| {
            let mut busy = self.shared.busy.lock().unwrap_or_else(|e| e.into_inner());
            busy[slot] = busy[slot].saturating_sub(1);
            returned.0
        })
    }

    pub fn wait_slot(&self, slot: usize) {
        let mut busy = self.shared.busy.lock().unwrap_or_else(|e| e.into_inner());
        while busy[slot] > 0 {
            let (next, timeout) = self.shared.idle.wait_timeout(busy, Duration::from_secs(3)).unwrap_or_else(|e| e.into_inner());
            busy = next;
            if timeout.timed_out() {
                tracing::warn!("frame generation: the presenter thread did not finish a frame in 3 seconds");
                busy[slot] = 0;
            }
        }
    }

    pub fn flush(&self) {
        for slot in 0..FRAMES_IN_FLIGHT {
            self.wait_slot(slot);
        }
    }

    pub fn take_resize(&self) -> bool {
        self.shared.resize.swap(false, Ordering::Relaxed)
    }

    pub fn presented(&self) -> (u64, u64) {
        (self.shared.real.load(Ordering::Relaxed), self.shared.generated.load(Ordering::Relaxed))
    }
}

impl Drop for FramePacer {
    fn drop(&mut self) {
        self.jobs = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_are_free_until_a_frame_is_handed_over() {
        let shared = Shared { busy: Mutex::new([0; FRAMES_IN_FLIGHT]), idle: Condvar::new(), resize: AtomicBool::new(true), real: AtomicU64::new(0), generated: AtomicU64::new(0) };
        let pacer = FramePacer { jobs: None, shared: Arc::new(shared), thread: None };
        pacer.flush();
        assert!(pacer.take_resize());
        assert!(!pacer.take_resize());
        assert_eq!(pacer.presented(), (0, 0));
    }
}
