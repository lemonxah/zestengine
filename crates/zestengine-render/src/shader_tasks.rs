//! Owned shader recipes run on a few compiler threads. Polling never waits for
//! the driver. Owners must drain their jobs before destroying referenced Vulkan objects.
//! Jobs known to be small have a thread of their own ([`Lane::Quick`]), so they
//! finish while the compiler threads are minutes deep in material variants.
use std::{collections::HashMap, sync::{Arc, Condvar, Mutex, OnceLock, mpsc}, time::Instant};

type Work = Box<dyn FnOnce() + Send>;
static WORKER: OnceLock<mpsc::Sender<Work>> = OnceLock::new();
static QUICK: OnceLock<mpsc::Sender<Work>> = OnceLock::new();

/// Which threads run a job.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lane {
    /// The shared compiler threads, in queue order.
    Compiler,
    /// One more thread for jobs that take milliseconds; never behind `Compiler`'s queue.
    Quick,
}

#[cfg(all(unix, not(target_vendor = "apple")))]
fn lower_thread_priority() {
    unsafe extern "C" { fn setpriority(which: i32, who: u32, priority: i32) -> i32; }
    unsafe { setpriority(0, 0, 19); }
}

#[cfg(windows)]
fn lower_thread_priority() {
    unsafe extern "system" {
        fn GetCurrentThread() -> isize;
        fn SetThreadPriority(thread: isize, priority: i32) -> i32;
    }
    unsafe { SetThreadPriority(GetCurrentThread(), -2); }
}

#[cfg(target_vendor = "apple")]
fn lower_thread_priority() {
    unsafe extern "C" { fn pthread_set_qos_class_self_np(class: u32, relative_priority: i32) -> i32; }
    unsafe { pthread_set_qos_class_self_np(0x11, 0); }
}

#[cfg(not(any(unix, windows)))]
fn lower_thread_priority() {}

fn spawn(name: String, rx: Arc<Mutex<mpsc::Receiver<Work>>>, background: bool) {
    std::thread::Builder::new().name(name).spawn(move || {
        if background { lower_thread_priority(); }
        loop {
            // The queue lock is released before the job runs.
            let work = rx.lock().unwrap().recv();
            match work { Ok(work) => work(), Err(_) => break }
        }
    }).expect("starting shader compiler thread");
}
static TASKS: OnceLock<Mutex<HashMap<u64, Task>>> = OnceLock::new();
/// Jobs finished per device, including the ones [`poll`] has already handed out.
static FINISHED: Mutex<Vec<(u64, usize)>> = Mutex::new(Vec::new());
static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

struct Task { device: u64, label: String, started: Instant, compiling: bool, result: Option<Result<(), String>> }

/// A real compiler job, including queue time. Completed entries are consumed by
/// the editor; the pipeline result remains owned by the renderer.
#[derive(Clone, Debug)]
pub struct ShaderTask {
    pub id: u64,
    pub label: String,
    pub elapsed_seconds: u64,
    pub compiling: bool,
    pub result: Option<Result<(), String>>,
}

pub(crate) fn poll(device: u64) -> Vec<ShaderTask> {
    let mut tasks = TASKS.get_or_init(Default::default).lock().unwrap();
    let mut out = Vec::new();
    tasks.retain(|&id, t| {
        if t.device != device { return true; }
        out.push(ShaderTask { id, label: t.label.clone(), elapsed_seconds: t.started.elapsed().as_secs(),
            compiling: t.compiling, result: t.result.clone() });
        t.result.is_none()
    });
    out.sort_by_key(|t| t.id);
    out
}

pub(crate) fn forget(device: u64) {
    TASKS.get_or_init(Default::default).lock().unwrap().retain(|_, t| t.device != device);
    FINISHED.lock().unwrap().retain(|(d, _)| *d != device);
}

/// Where a device's compiler jobs stand. Reading it consumes nothing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ShaderProgress {
    /// Queued or compiling.
    pub pending: usize,
    /// Finished since the device was created, failures included.
    pub finished: usize,
    /// The longest-running job being compiled now.
    pub compiling: Option<String>,
}

pub(crate) fn progress(device: u64) -> ShaderProgress {
    let finished = FINISHED.lock().unwrap().iter().find(|(d, _)| *d == device).map_or(0, |(_, n)| *n);
    let tasks = TASKS.get_or_init(Default::default).lock().unwrap();
    let pending = || tasks.values().filter(|t| t.device == device && t.result.is_none());
    ShaderProgress {
        pending: pending().count(),
        finished,
        compiling: pending().filter(|t| t.compiling).min_by_key(|t| t.started).map(|t| t.label.clone()),
    }
}

fn finish(id: u64, device: u64, result: Result<(), String>) {
    if let Some(t) = TASKS.get().unwrap().lock().unwrap().get_mut(&id) { t.result = Some(result); }
    let mut finished = FINISHED.lock().unwrap();
    match finished.iter_mut().find(|(d, _)| *d == device) {
        Some((_, n)) => *n += 1,
        None => finished.push((device, 1)),
    }
}

fn workers() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

type Outcome<T> = Result<T, String>;
struct State<T> { value: Mutex<Option<Outcome<T>>>, ready: Condvar }
pub(crate) struct Job<T> { state: Arc<State<T>> }

impl<T: Send + 'static> Job<T> {
    pub fn start(device: u64, label: String, build: impl FnOnce() -> anyhow::Result<T> + Send + 'static) -> Self {
        Self::start_on(Lane::Compiler, device, label, build)
    }

    pub fn start_on(lane: Lane, device: u64, label: String, build: impl FnOnce() -> anyhow::Result<T> + Send + 'static) -> Self {
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        TASKS.get_or_init(Default::default).lock().unwrap().insert(id, Task {
            device, label: label.clone(), started: Instant::now(), compiling: false, result: None,
        });
        let state = Arc::new(State { value: Mutex::new(None), ready: Condvar::new() });
        let finished = state.clone();
        let work: Work = Box::new(move || {
            if let Some(t) = TASKS.get().unwrap().lock().unwrap().get_mut(&id) { t.compiling = true; }
            tracing::info!(%label, "background shader compilation started");
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(build))
                .map_err(|_| "shader compiler worker panicked".to_owned())
                .and_then(|r| r.map_err(|e| format!("{e:#}")));
            if let Err(e) = &result { tracing::error!(%label, error = %e, "shader compilation failed"); }
            tracing::info!(%label, success = result.is_ok(), "background shader compilation finished");
            let outcome = result.as_ref().map(|_| ()).map_err(Clone::clone);
            // Counted finished while the value is held: whoever sees one finds the other.
            let mut value = finished.value.lock().unwrap();
            *value = Some(result);
            finish(id, device, outcome);
            drop(value);
            finished.ready.notify_all();
        });
        // GPU tests render a fixed number of frames, and a view whose pipeline
        // is still queued draws nothing. Compile inline there. The sentinel
        // device ids belong to this module's own queue tests.
        #[cfg(test)]
        if device < u64::MAX - 3 { work(); return Self { state }; }
        let worker = match lane {
            Lane::Compiler => WORKER.get_or_init(|| {
                let (tx, rx) = mpsc::channel::<Work>();
                let rx = Arc::new(Mutex::new(rx));
                for index in 0..workers() { spawn(format!("shader-compiler-{index}"), rx.clone(), true); }
                tx
            }),
            Lane::Quick => QUICK.get_or_init(|| {
                let (tx, rx) = mpsc::channel::<Work>();
                spawn("shader-quick".into(), Arc::new(Mutex::new(rx)), false);
                tx
            }),
        };
        if worker.send(work).is_err() {
            let error = "shader compiler is unavailable".to_owned();
            let mut value = state.value.lock().unwrap();
            *value = Some(Err(error.clone()));
            finish(id, device, Err(error));
            drop(value);
            state.ready.notify_all();
        }
        Self { state }
    }

    /// No driver call, join, or cache lock on this path.
    pub fn take_ready(&self) -> Option<Outcome<T>> { self.state.value.lock().unwrap().take() }

    /// Teardown only: keep layouts/modules/device alive until compilation ends.
    pub fn finish(self) -> Outcome<T> {
        let mut value = self.state.value.lock().unwrap();
        while value.is_none() { value = self.state.ready.wait(value).unwrap(); }
        value.take().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn blocked_compiler_does_not_block_polling_and_reports_completion() {
        let (entered, entry) = mpsc::channel();
        let (release, gate) = mpsc::channel();
        let job = Job::start(u64::MAX, "test shader".into(), move || {
            entered.send(()).unwrap(); gate.recv().unwrap(); Ok(42)
        });
        entry.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        assert!(job.take_ready().is_none());
        let tasks = poll(u64::MAX);
        assert_eq!(tasks.len(), 1);
        assert!(tasks[0].compiling && tasks[0].result.is_none());
        assert_eq!(progress(u64::MAX), ShaderProgress { pending: 1, finished: 0, compiling: Some("test shader".into()) });
        release.send(()).unwrap();
        assert_eq!(job.finish().unwrap(), 42);
        assert_eq!(progress(u64::MAX), ShaderProgress { pending: 0, finished: 1, compiling: None });
        assert_eq!(poll(u64::MAX)[0].result, Some(Ok(())));
        assert!(poll(u64::MAX).is_empty());
        // Handing the result out does not uncount it.
        assert_eq!(progress(u64::MAX).finished, 1);
        forget(u64::MAX);
        assert_eq!(progress(u64::MAX), ShaderProgress::default());
    }

    #[test]
    fn a_quick_job_does_not_wait_for_the_compiler_queue() {
        // Every compiler thread, and one more job behind them, is held up.
        let (release, gate) = mpsc::channel::<()>();
        let gate = Arc::new(Mutex::new(gate));
        let held: Vec<_> = (0..workers() + 1).map(|_| {
            let gate = gate.clone();
            Job::start(u64::MAX - 3, "held".into(), move || { gate.lock().unwrap().recv().unwrap(); Ok(()) })
        }).collect();
        assert_eq!(Job::start_on(Lane::Quick, u64::MAX - 3, "quick".into(), || Ok(5)).finish().unwrap(), 5);
        for _ in 0..held.len() { release.send(()).unwrap(); }
        for job in held { job.finish().unwrap(); }
        forget(u64::MAX - 3);
    }

    #[test]
    fn jobs_compile_side_by_side() {
        if workers() < 2 { return; }
        let (entered, entry) = mpsc::channel();
        let gates: Vec<_> = (0..2).map(|_| {
            let (release, gate) = mpsc::channel::<()>();
            let entered = entered.clone();
            (release, Job::start(u64::MAX - 2, "side by side".into(), move || { entered.send(()).unwrap(); gate.recv().unwrap(); Ok(()) }))
        }).collect();
        // Both are inside their build before either is let go.
        for _ in 0..2 { entry.recv_timeout(std::time::Duration::from_secs(5)).unwrap(); }
        for (release, job) in gates { release.send(()).unwrap(); job.finish().unwrap(); }
        forget(u64::MAX - 2);
    }

    #[test]
    fn failures_and_panics_finish_tasks_without_killing_the_worker() {
        let failed = Job::<()>::start(u64::MAX-1, "failure".into(), || anyhow::bail!("bad shader"));
        assert!(failed.finish().unwrap_err().contains("bad shader"));
        let panicked = Job::<()>::start(u64::MAX-1, "panic".into(), || panic!("test panic"));
        assert!(panicked.finish().unwrap_err().contains("panicked"));
        assert!(poll(u64::MAX-1).iter().all(|t| matches!(t.result, Some(Err(_)))));
        assert_eq!(Job::start(u64::MAX-1, "next".into(), || Ok(7)).finish().unwrap(), 7);
        forget(u64::MAX-1);
    }
}
