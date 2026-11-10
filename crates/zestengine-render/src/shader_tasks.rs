//! Owned shader recipes run on one compiler thread. Polling never waits for the
//! driver. Owners must drain their jobs before destroying referenced Vulkan objects.
use std::{collections::HashMap, sync::{Arc, Condvar, Mutex, OnceLock, mpsc}, time::Instant};

type Work = Box<dyn FnOnce() + Send>;
static WORKER: OnceLock<mpsc::Sender<Work>> = OnceLock::new();
static TASKS: OnceLock<Mutex<HashMap<u64, Task>>> = OnceLock::new();
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
}

type Outcome<T> = Result<T, String>;
struct State<T> { value: Mutex<Option<Outcome<T>>>, ready: Condvar }
pub(crate) struct Job<T> { state: Arc<State<T>> }

impl<T: Send + 'static> Job<T> {
    pub fn start(device: u64, label: String, build: impl FnOnce() -> anyhow::Result<T> + Send + 'static) -> Self {
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
            if let Some(t) = TASKS.get().unwrap().lock().unwrap().get_mut(&id) {
                t.result = Some(result.as_ref().map(|_| ()).map_err(Clone::clone));
            }
            tracing::info!(%label, success = result.is_ok(), "background shader compilation finished");
            *finished.value.lock().unwrap() = Some(result);
            finished.ready.notify_all();
        });
        // GPU tests render a fixed number of frames, and a view whose pipeline
        // is still queued draws nothing. Compile inline there. The sentinel
        // device ids belong to this module's own queue tests.
        #[cfg(test)]
        if device < u64::MAX - 1 { work(); return Self { state }; }
        let worker = WORKER.get_or_init(|| {
            let (tx, rx) = mpsc::channel::<Work>();
            std::thread::Builder::new().name("shader-compiler".into()).spawn(move || {
                while let Ok(work) = rx.recv() { work(); }
            }).expect("starting shader compiler thread");
            tx
        });
        if worker.send(work).is_err() {
            let error = "shader compiler is unavailable".to_owned();
            TASKS.get().unwrap().lock().unwrap().get_mut(&id).unwrap().result = Some(Err(error.clone()));
            *state.value.lock().unwrap() = Some(Err(error));
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
        release.send(()).unwrap();
        assert_eq!(job.finish().unwrap(), 42);
        assert_eq!(poll(u64::MAX)[0].result, Some(Ok(())));
        assert!(poll(u64::MAX).is_empty());
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
