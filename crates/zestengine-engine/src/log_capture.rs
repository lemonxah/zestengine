//! Logging: the global `tracing` subscriber and the editor's in-app console.
//!
//! [`init`] installs stdout formatting plus, in the editor, a capture layer
//! that mirrors events into a ring buffer the Log tab reads ([`store`]). A game
//! build has no Log tab, so it installs no capture layer.
//!
//! Neither path may stall the thread that emits an event (often the main
//! thread), even during a flood of thousands of events per second:
//! - stdout goes through a bounded, lossy channel to a writer thread
//!   ([`pipe`]), so a slow terminal or pipe drops lines instead of blocking;
//!   it's flushed at process exit and before a panic message.
//! - the capture layer formats outside the ring lock, folds consecutive
//!   identical messages into one row with a repeat count, and stores at most
//!   [`CAPTURE_BUDGET`] new rows per second (the rest are counted into a
//!   "… N more messages suppressed" row).
//! - the Log tab's filtered view ([`LogView`]) extends itself incrementally
//!   when entries were only appended.

use tracing_subscriber::prelude::*;

pub use capture::*;

/// Install the global tracing subscriber: stdout formatting (honouring
/// `RUST_LOG`, default `info`) plus, in the editor, the in-app capture layer.
/// Call once at startup in place of the bare `tracing_subscriber::fmt().init()`.
pub fn init() {
    // `calloop` spams benign "event for non-existence source" WARNs during
    // cursor-grab churn on Wayland, and `sctk_adwaita` (the Wayland title bar)
    // warns about desktop button layouts it doesn't draw (e.g. `icon`); cap
    // both at error so they stay quiet even when RUST_LOG raises the level.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "info".into())
        .add_directive("calloop=error".parse().expect("valid directive"))
        .add_directive("sctk_adwaita=error".parse().expect("valid directive"));
    // Android has no stdout: lines go to logcat (`adb logcat -s zestengine`).
    #[cfg(target_os = "android")]
    let fmt = tracing_subscriber::fmt::layer().with_ansi(false).with_writer(crate::runtime::Logcat);
    #[cfg(not(target_os = "android"))]
    let fmt = {
        pipe::install_exit_flush();
        tracing_subscriber::fmt::layer().with_writer(pipe::Stdout)
    };
    let registry = tracing_subscriber::registry().with(filter).with(fmt);
    registry.with(CaptureLayer::global()).init();
}

/// Non-blocking line writer: the fmt layer hands each formatted event to a
/// bounded channel (never waiting; a full channel drops the line and counts
/// it) and a writer thread batches them into the sink.
#[cfg_attr(target_os = "android", allow(dead_code))]
mod pipe {
    use std::io::{self, Write};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
    use std::sync::{Arc, OnceLock};
    use std::time::{Duration, Instant};

    use tracing_subscriber::fmt::MakeWriter;

    /// Lines queued for the stdout thread before new ones are dropped.
    pub const STDOUT_QUEUE: usize = 16 * 1024;
    /// Bytes gathered per sink write.
    const BATCH_BYTES: usize = 256 * 1024;

    enum Msg {
        Line(Vec<u8>),
        /// Acked once everything queued before it has been written.
        Flush(mpsc::Sender<()>),
    }

    #[derive(Clone)]
    pub struct LinePipe {
        tx: SyncSender<Msg>,
        dropped: Arc<AtomicU64>,
    }

    impl LinePipe {
        /// Start the writer thread over `sink`, buffering up to `capacity` lines.
        pub fn spawn(name: &str, sink: impl Write + Send + 'static, capacity: usize) -> io::Result<Self> {
            let (tx, rx) = mpsc::sync_channel(capacity);
            let dropped = Arc::new(AtomicU64::new(0));
            let d = dropped.clone();
            std::thread::Builder::new().name(name.to_owned()).spawn(move || run(rx, sink, &d))?;
            Ok(Self { tx, dropped })
        }

        /// Queue one chunk without waiting; a full queue drops it.
        fn send(&self, buf: &[u8]) {
            if let Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) = self.tx.try_send(Msg::Line(buf.to_vec())) {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }

        /// Wait (up to `timeout`) until every line queued so far is written.
        pub fn flush(&self, timeout: Duration) -> bool {
            let deadline = Instant::now() + timeout;
            let (ack_tx, ack_rx) = mpsc::channel();
            let mut msg = Msg::Flush(ack_tx);
            loop {
                match self.tx.try_send(msg) {
                    Ok(()) => break,
                    Err(TrySendError::Full(m)) if Instant::now() < deadline => {
                        msg = m;
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    Err(_) => return false,
                }
            }
            ack_rx.recv_timeout(deadline.saturating_duration_since(Instant::now())).is_ok()
        }
    }

    fn run(rx: Receiver<Msg>, mut sink: impl Write, dropped: &AtomicU64) {
        let mut batch = Vec::with_capacity(64 * 1024);
        let mut acks = Vec::new();
        while let Ok(first) = rx.recv() {
            let mut next = Some(first);
            while let Some(m) = next {
                match m {
                    Msg::Line(l) => batch.extend_from_slice(&l),
                    Msg::Flush(ack) => acks.push(ack),
                }
                next = if batch.len() < BATCH_BYTES { rx.try_recv().ok() } else { None };
            }
            let n = dropped.swap(0, Ordering::Relaxed);
            if n > 0 {
                let _ = writeln!(batch, "[zestengine] {n} log lines dropped: output could not keep up");
            }
            if !batch.is_empty() {
                let _ = sink.write_all(&batch);
                let _ = sink.flush();
                batch.clear();
                if batch.capacity() > 4 * BATCH_BYTES {
                    batch.shrink_to(64 * 1024);
                }
            }
            for a in acks.drain(..) {
                let _ = a.send(());
            }
        }
    }

    pub struct PipeWriter<'a>(&'a LinePipe);

    impl Write for PipeWriter<'_> {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if !buf.is_empty() {
                self.0.send(buf);
            }
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for LinePipe {
        type Writer = PipeWriter<'a>;
        fn make_writer(&'a self) -> PipeWriter<'a> {
            PipeWriter(self)
        }
    }

    /// The process-wide stdout pipe (`None` if its thread couldn't start).
    static STDOUT: OnceLock<Option<LinePipe>> = OnceLock::new();

    fn stdout_pipe() -> Option<&'static LinePipe> {
        STDOUT.get_or_init(|| LinePipe::spawn("zest-log-stdout", io::stdout(), STDOUT_QUEUE).ok()).as_ref()
    }

    /// `MakeWriter` for the fmt layer: the non-blocking pipe, or plain stdout
    /// if the writer thread couldn't be spawned.
    pub struct Stdout;

    pub enum StdoutWriter<'a> {
        Pipe(PipeWriter<'a>),
        Direct(io::Stdout),
    }

    impl Write for StdoutWriter<'_> {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            match self {
                Self::Pipe(w) => w.write(buf),
                Self::Direct(w) => w.write(buf),
            }
        }
        fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
            match self {
                Self::Pipe(w) => w.write_all(buf),
                Self::Direct(w) => w.write_all(buf),
            }
        }
        fn flush(&mut self) -> io::Result<()> {
            match self {
                Self::Pipe(w) => w.flush(),
                Self::Direct(w) => w.flush(),
            }
        }
    }

    impl<'a> MakeWriter<'a> for Stdout {
        type Writer = StdoutWriter<'a>;
        fn make_writer(&'a self) -> StdoutWriter<'a> {
            match stdout_pipe() {
                Some(p) => StdoutWriter::Pipe(PipeWriter(p)),
                None => StdoutWriter::Direct(io::stdout()),
            }
        }
    }

    /// Write out every stdout log line queued so far (waits at most `timeout`).
    pub fn flush_stdout(timeout: Duration) {
        if let Some(Some(p)) = STDOUT.get() {
            p.flush(timeout);
        }
    }

    /// Keep the pipe alive for the whole process and drain it on the way out:
    /// at normal exit (`atexit`, which also covers `std::process::exit`) and
    /// before a panic message, so the last lines aren't lost or misordered.
    pub fn install_exit_flush() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            if stdout_pipe().is_none() {
                return;
            }
            #[cfg(any(unix, windows))]
            {
                extern "C" fn flush_at_exit() {
                    flush_stdout(Duration::from_secs(1));
                }
                // SAFETY: registering a plain `extern "C"` callback.
                unsafe {
                    libc::atexit(flush_at_exit);
                }
            }
            let previous = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                flush_stdout(Duration::from_millis(250));
                previous(info);
            }));
        });
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::sync::Mutex;

        /// A sink that takes 20 ms per write, like a terminal that can't keep up.
        struct SlowSink(Arc<Mutex<Vec<u8>>>);
        impl Write for SlowSink {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                std::thread::sleep(Duration::from_millis(20));
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        #[test]
        fn slow_sink_never_blocks_the_emitter() {
            let out = Arc::new(Mutex::new(Vec::new()));
            let pipe = LinePipe::spawn("test-log-pipe", SlowSink(out.clone()), 8).unwrap();
            let t = Instant::now();
            for i in 0..2000 {
                pipe.make_writer().write_all(format!("line {i}\n").as_bytes()).unwrap();
            }
            // A blocking writer would need ~2000/8 * 20 ms here.
            assert!(t.elapsed() < Duration::from_millis(500), "emitter blocked: {:?}", t.elapsed());
            assert!(pipe.flush(Duration::from_secs(10)));
            let text = String::from_utf8(out.lock().unwrap().clone()).unwrap();
            let written = text.lines().filter(|l| l.starts_with("line ")).count() as u64;
            let dropped: u64 = text
                .lines()
                .filter_map(|l| l.strip_prefix("[zestengine] "))
                .map(|l| l.split(' ').next().unwrap().parse::<u64>().unwrap())
                .sum();
            assert!(written > 0 && dropped > 0);
            assert_eq!(written + dropped, 2000, "every line is written or counted as dropped");
            // Order is preserved for what got through.
            let nums: Vec<u64> =
                text.lines().filter_map(|l| l.strip_prefix("line ")).map(|n| n.parse().unwrap()).collect();
            assert!(nums.windows(2).all(|w| w[0] < w[1]));
        }
    }
}

/// In-editor capture: the ring the Log tab shows, the tracing layer feeding
/// it, and the Log tab's incrementally maintained filtered view.
mod capture {
    use std::collections::VecDeque;
    use std::fmt::Write as _;
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
    use std::sync::{Arc, Mutex, OnceLock, PoisonError};
    use std::time::Instant;

    use tracing::Level;
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::Context;

    /// Most entries the Log tab keeps (oldest dropped first).
    pub const LOG_CAP: usize = 1000;
    /// New rows the capture stores per second; past it events are only
    /// counted, then reported as one summary row (stdout still gets them).
    pub const CAPTURE_BUDGET: u32 = 500;
    const WINDOW_MS: u64 = 1000;

    /// Local wall-clock time of day, in milliseconds (shown "HH:MM:SS.mmm").
    #[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
    pub struct LogTime(u32);

    impl LogTime {
        fn now() -> Self {
            use chrono::Timelike;
            let t = chrono::Local::now().time();
            Self(t.num_seconds_from_midnight() * 1000 + (t.nanosecond() / 1_000_000).min(999))
        }
    }

    impl std::fmt::Display for LogTime {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let ms = self.0;
            write!(f, "{:02}:{:02}:{:02}.{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
        }
    }

    /// One captured log record (or a run of identical consecutive ones).
    pub struct LogEntry {
        pub level: Level,
        pub target: &'static str,
        pub message: String,
        /// Time of the latest occurrence.
        pub time: LogTime,
        /// How many identical events in a row this row stands for (>= 1).
        pub repeat: u32,
    }

    /// Ring buffer of recent log entries; oldest dropped past the cap.
    pub struct LogRing {
        pub entries: VecDeque<LogEntry>,
        cap: usize,
        /// Sequence number of `entries[0]` (monotonic; stable ids for cached views).
        pub first_seq: u64,
        /// Bumped on every change (push, repeat, clear) so views can cache against it.
        pub generation: u64,
        /// Warning / error rows currently in the ring.
        pub warns: usize,
        pub errors: usize,
    }

    impl LogRing {
        pub fn new(cap: usize) -> Self {
            Self {
                entries: VecDeque::with_capacity(cap.max(1)),
                cap: cap.max(1),
                first_seq: 0,
                generation: 0,
                warns: 0,
                errors: 0,
            }
        }

        /// The entry with ring sequence number `seq`, if still held.
        pub fn get(&self, seq: u64) -> Option<&LogEntry> {
            self.entries.get(usize::try_from(seq.checked_sub(self.first_seq)?).ok()?)
        }

        /// One past the newest entry's sequence number.
        pub fn next_seq(&self) -> u64 {
            self.first_seq + self.entries.len() as u64
        }

        fn count(&mut self, level: Level, add: bool) {
            let c = match level {
                Level::WARN => &mut self.warns,
                Level::ERROR => &mut self.errors,
                _ => return,
            };
            if add { *c += 1 } else { *c = c.saturating_sub(1) }
        }

        /// Append `entry`, or fold it into the newest entry when level, target
        /// and message match. Never allocates (capacity is reserved up front);
        /// returns what should be dropped once the lock is released.
        pub fn push(&mut self, entry: LogEntry) -> Option<LogEntry> {
            self.generation += 1;
            if let Some(last) = self.entries.back_mut()
                && last.level == entry.level
                && last.target == entry.target
                && last.message == entry.message
            {
                last.repeat = last.repeat.saturating_add(entry.repeat);
                last.time = entry.time;
                return Some(entry);
            }
            let evicted = if self.entries.len() >= self.cap {
                let old = self.entries.pop_front();
                if let Some(o) = &old {
                    self.first_seq += 1;
                    self.count(o.level, false);
                }
                old
            } else {
                None
            };
            self.count(entry.level, true);
            self.entries.push_back(entry);
            evicted
        }

        pub fn clear(&mut self) {
            self.first_seq += self.entries.len() as u64;
            self.entries.clear();
            self.warns = 0;
            self.errors = 0;
            self.generation += 1;
        }
    }

    /// Lock-free per-window budget: the first `budget` events of each window
    /// pass, the rest are counted. Checked before an event is even formatted.
    pub struct RateLimiter {
        budget: u32,
        window_ms: u64,
        window_start: AtomicU64,
        count: AtomicU32,
        suppressed: AtomicU64,
    }

    impl RateLimiter {
        pub fn new(budget: u32, window_ms: u64) -> Self {
            Self {
                budget,
                window_ms: window_ms.max(1),
                window_start: AtomicU64::new(0),
                count: AtomicU32::new(0),
                suppressed: AtomicU64::new(0),
            }
        }

        /// Start a new window if the current one has ended; returns how many
        /// events the ended window suppressed (if any).
        pub fn roll(&self, now_ms: u64) -> Option<u64> {
            let start = self.window_start.load(Ordering::Acquire);
            if now_ms.saturating_sub(start) < self.window_ms {
                return None;
            }
            // One thread wins the rollover and reports the ended window.
            self.window_start.compare_exchange(start, now_ms, Ordering::AcqRel, Ordering::Relaxed).ok()?;
            self.count.store(0, Ordering::Release);
            Some(self.suppressed.swap(0, Ordering::AcqRel)).filter(|&n| n > 0)
        }

        /// Whether to store this event, plus the ended window's suppressed
        /// count when this call rolled the window over.
        pub fn admit(&self, now_ms: u64) -> (bool, Option<u64>) {
            let summary = self.roll(now_ms);
            if self.count.fetch_add(1, Ordering::AcqRel) < self.budget {
                (true, summary)
            } else {
                self.suppressed.fetch_add(1, Ordering::Relaxed);
                (false, summary)
            }
        }
    }

    /// The ring plus its rate limiter.
    pub struct Capture {
        ring: Mutex<LogRing>,
        limiter: RateLimiter,
        start: Instant,
    }

    impl Capture {
        pub fn new(cap: usize, budget: u32) -> Self {
            Self { ring: Mutex::new(LogRing::new(cap)), limiter: RateLimiter::new(budget, WINDOW_MS), start: Instant::now() }
        }

        fn now_ms(&self) -> u64 {
            self.start.elapsed().as_millis() as u64
        }

        /// The critical section: a compare and a `push_back`; anything evicted
        /// or folded is dropped after the guard is released.
        fn push(&self, entry: LogEntry) {
            let spare = self.ring.lock().unwrap_or_else(PoisonError::into_inner).push(entry);
            drop(spare);
        }

        fn push_summary(&self, n: u64) {
            self.push(LogEntry {
                level: Level::WARN,
                target: "zestengine::log",
                message: format!("… {n} more messages suppressed this second (log flood; stdout still has them)"),
                time: LogTime::now(),
                repeat: 1,
            });
        }

        /// Report a finished flood window even when no further event arrives.
        pub fn poll(&self) {
            self.poll_at(self.now_ms());
        }

        fn poll_at(&self, now_ms: u64) {
            if let Some(n) = self.limiter.roll(now_ms) {
                self.push_summary(n);
            }
        }

        fn record(&self, event: &tracing::Event<'_>) {
            let (admit, summary) = self.limiter.admit(self.now_ms());
            if let Some(n) = summary {
                self.push_summary(n);
            }
            if !admit {
                return;
            }
            let meta = event.metadata();
            let mut visitor = MessageVisitor::default();
            event.record(&mut visitor);
            self.push(LogEntry {
                level: *meta.level(),
                target: meta.target(),
                message: visitor.finish(),
                time: LogTime::now(),
                repeat: 1,
            });
        }
    }

    static GLOBAL: OnceLock<Arc<Capture>> = OnceLock::new();

    fn global() -> &'static Arc<Capture> {
        GLOBAL.get_or_init(|| Arc::new(Capture::new(LOG_CAP, CAPTURE_BUDGET)))
    }

    /// The shared log ring (created on first use). Hold the lock briefly:
    /// copy what you need out, then lay out without it.
    pub fn store() -> &'static Mutex<LogRing> {
        &global().ring
    }

    /// Flush a pending "suppressed" summary (call once per frame).
    pub fn poll() {
        global().poll();
    }

    /// Tracing layer that appends each event to a capture ring.
    pub struct CaptureLayer(Arc<Capture>);

    impl CaptureLayer {
        pub fn global() -> Self {
            Self(global().clone())
        }
    }

    impl<S: tracing::Subscriber> Layer<S> for CaptureLayer {
        fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
            self.0.record(event);
        }
    }

    /// Pulls the `message` field out of an event and appends any other fields as
    /// `key=value`, so a structured `tracing::info!(gpu = ?name, "selected")` reads
    /// as `selected gpu=...`.
    #[derive(Default)]
    struct MessageVisitor {
        message: String,
        fields: String,
    }

    impl MessageVisitor {
        fn finish(mut self) -> String {
            if !self.fields.is_empty() {
                if !self.message.is_empty() {
                    self.message.push(' ');
                }
                self.message.push_str(&self.fields);
            }
            self.message
        }
    }

    impl tracing::field::Visit for MessageVisitor {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            if field.name() == "message" {
                let _ = write!(self.message, "{value:?}");
            } else {
                if !self.fields.is_empty() {
                    self.fields.push(' ');
                }
                let _ = write!(self.fields, "{}={value:?}", field.name());
            }
        }
    }

    /// Bit for `level` in a [`LogView`] level mask.
    pub fn level_bit(level: Level) -> u8 {
        match level {
            Level::ERROR => 1,
            Level::WARN => 2,
            Level::INFO => 4,
            Level::DEBUG => 8,
            Level::TRACE => 16,
        }
    }

    /// One shown log entry: its ring sequence number and the char count of
    /// each physical line of its message (for the virtualised row heights).
    #[derive(Debug, PartialEq)]
    pub struct ViewItem {
        pub seq: u64,
        pub line_lens: Vec<u32>,
    }

    /// The Log tab's filtered view of the ring. [`LogView::sync`] re-filters the
    /// whole ring only when the filter changes; otherwise it drops evicted
    /// entries and scans just the ones appended since the last sync.
    #[derive(Default)]
    pub struct LogView {
        pub items: Vec<ViewItem>,
        /// (index into `items`, line within the entry), one per physical line.
        pub lines: Vec<(u32, u32)>,
        built: bool,
        levels: u8,
        search: String,
        /// `search`, lowercased.
        needle: String,
        generation: u64,
        /// One past the last ring sequence number scanned.
        scanned: u64,
        /// Whole-ring re-filters so far (tests check appends stay incremental).
        #[cfg_attr(not(test), allow(dead_code))]
        full_rebuilds: u64,
    }

    impl LogView {
        /// Bring the view up to date with `ring` for the `levels` mask (see
        /// [`level_bit`]) and the case-insensitive `search` text. Returns
        /// whether anything was recomputed.
        pub fn sync(&mut self, ring: &LogRing, levels: u8, search: &str) -> bool {
            let same_filter = self.built && self.levels == levels && self.search == search;
            if same_filter && self.generation == ring.generation {
                return false;
            }
            if same_filter {
                // Drop entries the ring evicted (or cleared) since last time.
                let gone = self.items.partition_point(|it| it.seq < ring.first_seq);
                if gone > 0 {
                    self.items.drain(..gone);
                    let gone_lines = self.lines.partition_point(|&(i, _)| (i as usize) < gone);
                    self.lines.drain(..gone_lines);
                    for l in &mut self.lines {
                        l.0 -= gone as u32;
                    }
                }
            } else {
                self.built = true;
                self.levels = levels;
                if self.search != search {
                    self.search.clear();
                    self.search.push_str(search);
                    self.needle = search.to_lowercase();
                }
                self.items.clear();
                self.lines.clear();
                self.scanned = ring.first_seq;
                self.full_rebuilds += 1;
            }
            let skip = self.scanned.saturating_sub(ring.first_seq) as usize;
            for (i, e) in ring.entries.iter().enumerate().skip(skip) {
                if !matches(e, self.levels, &self.needle) {
                    continue;
                }
                let mut line_lens: Vec<u32> = e.message.lines().map(|l| l.chars().count() as u32).collect();
                if line_lens.is_empty() {
                    line_lens.push(0);
                }
                let idx = self.items.len() as u32;
                self.lines.extend((0..line_lens.len() as u32).map(|l| (idx, l)));
                self.items.push(ViewItem { seq: ring.first_seq + i as u64, line_lens });
            }
            self.scanned = ring.next_seq();
            self.generation = ring.generation;
            true
        }
    }

    fn matches(e: &LogEntry, levels: u8, needle: &str) -> bool {
        levels & level_bit(e.level) != 0
            && (needle.is_empty() || contains_ci(&e.message, needle) || contains_ci(e.target, needle))
    }

    /// Case-insensitive substring test against an already-lowercased needle;
    /// allocation-free for an ASCII needle (the common case).
    fn contains_ci(hay: &str, needle_lower: &str) -> bool {
        if needle_lower.is_ascii() {
            let (h, n) = (hay.as_bytes(), needle_lower.as_bytes());
            n.is_empty() || (n.len() <= h.len() && h.windows(n.len()).any(|w| w.eq_ignore_ascii_case(n)))
        } else {
            hay.to_lowercase().contains(needle_lower)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::time::Duration;

        fn entry(level: Level, msg: &str) -> LogEntry {
            LogEntry { level, target: "zestengine::test", message: msg.to_owned(), time: LogTime::default(), repeat: 1 }
        }

        const ALL: u8 = 31;

        #[test]
        fn consecutive_identical_messages_collapse() {
            let mut r = LogRing::new(10);
            for _ in 0..3 {
                r.push(entry(Level::WARN, "spam"));
            }
            r.push(entry(Level::INFO, "spam")); // different level: new row
            r.push(entry(Level::WARN, "spam")); // not consecutive: new row
            r.push(entry(Level::WARN, "spam"));
            let rows: Vec<_> = r.entries.iter().map(|e| (e.level, e.repeat)).collect();
            assert_eq!(rows, [(Level::WARN, 3), (Level::INFO, 1), (Level::WARN, 2)]);
            assert_eq!(r.warns, 2, "warn rows, not occurrences");
            assert_eq!(r.generation, 6, "every push is a change");
            let mut other_target = entry(Level::WARN, "spam");
            other_target.target = "other";
            r.push(other_target);
            assert_eq!(r.entries.len(), 4);
        }

        #[test]
        fn ring_evicts_oldest_and_keeps_counts() {
            let mut r = LogRing::new(3);
            for i in 0..5 {
                r.push(entry(if i % 2 == 0 { Level::ERROR } else { Level::INFO }, &format!("m{i}")));
            }
            assert_eq!(r.first_seq, 2);
            assert_eq!(r.next_seq(), 5);
            assert_eq!(r.get(2).unwrap().message, "m2");
            assert!(r.get(1).is_none());
            assert_eq!(r.errors, 2);
        }

        /// The incremental result must equal a fresh full rebuild.
        fn assert_matches_fresh(v: &LogView, r: &LogRing, levels: u8, search: &str) {
            let mut fresh = LogView::default();
            fresh.sync(r, levels, search);
            assert_eq!(v.items, fresh.items);
            assert_eq!(v.lines, fresh.lines);
        }

        #[test]
        fn view_extends_incrementally() {
            let mut r = LogRing::new(6);
            let mut v = LogView::default();
            r.push(entry(Level::INFO, "a\nb"));
            r.push(entry(Level::WARN, "c"));
            assert!(v.sync(&r, ALL, ""));
            assert_eq!(v.full_rebuilds, 1);
            assert!(!v.sync(&r, ALL, ""), "nothing changed");

            // Appends only: no full rebuild.
            for i in 0..3 {
                r.push(entry(Level::INFO, &format!("line {i}\nsecond")));
            }
            assert!(v.sync(&r, ALL, ""));
            assert_eq!(v.full_rebuilds, 1);
            assert_eq!(v.items.len(), 5);
            assert_matches_fresh(&v, &r, ALL, "");

            // A repeat changes the generation but adds nothing.
            r.push(entry(Level::INFO, "line 2\nsecond"));
            assert!(v.sync(&r, ALL, ""));
            assert_eq!((v.items.len(), v.full_rebuilds), (5, 1));

            // Wrap well past the cached first seq: still incremental.
            for i in 0..9 {
                r.push(entry(if i % 3 == 0 { Level::ERROR } else { Level::DEBUG }, &format!("w{i}\nx\ny")));
            }
            assert!(v.sync(&r, ALL, ""));
            assert_eq!(v.full_rebuilds, 1);
            assert_eq!(v.items.first().unwrap().seq, r.first_seq);
            assert_matches_fresh(&v, &r, ALL, "");

            // Filter change: one full rebuild, then incremental again.
            let errors = level_bit(Level::ERROR);
            v.sync(&r, errors, "W");
            assert_eq!(v.full_rebuilds, 2);
            assert_matches_fresh(&v, &r, errors, "W");
            assert!(v.items.iter().all(|it| r.get(it.seq).unwrap().level == Level::ERROR));
            r.push(entry(Level::ERROR, "w-new"));
            r.push(entry(Level::ERROR, "unrelated"));
            v.sync(&r, errors, "W");
            assert_eq!(v.full_rebuilds, 2);
            assert_matches_fresh(&v, &r, errors, "W");
            assert_eq!(r.get(v.items.last().unwrap().seq).unwrap().message, "w-new");

            // Clear empties the view without a rebuild.
            r.clear();
            v.sync(&r, errors, "W");
            assert!(v.items.is_empty() && v.lines.is_empty());
            r.push(entry(Level::ERROR, "WOW"));
            v.sync(&r, errors, "W");
            assert_eq!(v.items.len(), 1);
            assert_eq!(v.full_rebuilds, 2);
        }

        #[test]
        fn search_is_case_insensitive() {
            assert!(contains_ci("Shader COMPILE failed", "compile"));
            assert!(contains_ci("zestengine::render", "render"));
            assert!(!contains_ci("abc", "abcd"));
            assert!(contains_ci("Größe ÜBER", "über"));
            assert!(contains_ci("anything", ""));
        }

        #[test]
        fn rate_limiter_counts_and_reports_once_per_window() {
            let l = RateLimiter::new(3, 1000);
            let got: Vec<_> = (0..5).map(|_| l.admit(10)).collect();
            assert_eq!(got, [(true, None), (true, None), (true, None), (false, None), (false, None)]);
            assert_eq!(l.admit(999), (false, None), "same window");
            // Next window: the summary comes with the first event, and the budget resets.
            assert_eq!(l.admit(1000), (true, Some(3)));
            assert_eq!(l.roll(1500), None);
            assert_eq!(l.admit(1500), (true, None));
            // A quiet window reports nothing.
            assert_eq!(l.roll(5000), None);
        }

        fn dispatch(cap: &Arc<Capture>) -> tracing::Dispatch {
            tracing::Dispatch::new(tracing_subscriber::registry().with(CaptureLayer(cap.clone())))
        }

        use tracing_subscriber::prelude::*;

        #[test]
        fn layer_collapses_a_per_frame_warning() {
            let cap = Arc::new(Capture::new(100, 1000));
            tracing::dispatcher::with_default(&dispatch(&cap), || {
                for _ in 0..240 {
                    tracing::warn!(id = 7, "missing texture");
                }
                tracing::info!("done");
            });
            let r = cap.ring.lock().unwrap();
            let rows: Vec<_> = r.entries.iter().map(|e| (e.message.as_str(), e.repeat)).collect();
            assert_eq!(rows, [("missing texture id=7", 240), ("done", 1)]);
        }

        #[test]
        fn flood_is_summarised() {
            let cap = Arc::new(Capture::new(1000, 50));
            tracing::dispatcher::with_default(&dispatch(&cap), || {
                for i in 0..10_000 {
                    tracing::info!("flood {i}");
                }
            });
            // End the window: the suppressed remainder becomes one summary row.
            cap.poll_at(u64::MAX / 2);
            let r = cap.ring.lock().unwrap();
            let summary = |e: &LogEntry| {
                e.message.strip_prefix("… ")?.split(' ').next()?.parse::<u64>().ok().filter(|_| e.target == "zestengine::log")
            };
            let rows = r.entries.iter().filter(|e| summary(e).is_none()).count();
            assert!((50..=50 * 3).contains(&rows), "budgeted rows {rows}");
            let accounted: u64 = r.entries.iter().map(|e| summary(e).unwrap_or(e.repeat as u64)).sum();
            assert_eq!(accounted, 10_000, "every event is stored or counted");
            assert!(r.entries.back().and_then(summary).is_some());
        }

        /// 100k events from 4 worker threads while the main thread logs too:
        /// the main thread's per-event cost must stay small and bounded.
        fn flood_main_thread_cost(budget: u32) -> (Duration, Duration) {
            let cap = Arc::new(Capture::new(LOG_CAP, budget));
            let d = dispatch(&cap);
            let workers: Vec<_> = (0..4)
                .map(|w| {
                    let d = d.clone();
                    std::thread::spawn(move || {
                        tracing::dispatcher::with_default(&d, || {
                            for i in 0..25_000 {
                                tracing::info!(worker = w, "worker event {i}");
                            }
                        })
                    })
                })
                .collect();
            let mut samples = Vec::new();
            tracing::dispatcher::with_default(&d, || {
                let mut i = 0u64;
                while !workers.iter().all(|w| w.is_finished()) || samples.len() < 1000 {
                    let t = Instant::now();
                    tracing::warn!(frame = i, "main thread event");
                    samples.push(t.elapsed());
                    i += 1;
                    if samples.len() > 200_000 {
                        break;
                    }
                }
            });
            for w in workers {
                w.join().unwrap();
            }
            samples.sort();
            let mean = samples.iter().sum::<Duration>() / samples.len() as u32;
            let p99 = samples[samples.len() * 99 / 100];
            (mean, p99)
        }

        #[test]
        fn flood_keeps_main_thread_push_cheap() {
            let (mean, p99) = flood_main_thread_cost(CAPTURE_BUDGET);
            eprintln!("budgeted flood: main-thread mean {mean:?}, p99 {p99:?}");
            assert!(mean < Duration::from_micros(200), "mean {mean:?}");
            assert!(p99 < Duration::from_millis(2), "p99 {p99:?}");
        }

        /// Same without the budget (every event formatted and locked): the
        /// critical section alone must keep contention bounded.
        #[test]
        fn flood_unbudgeted_main_thread_cost() {
            let (mean, p99) = flood_main_thread_cost(u32::MAX);
            eprintln!("unbudgeted flood: main-thread mean {mean:?}, p99 {p99:?}");
            assert!(mean < Duration::from_micros(500), "mean {mean:?}");
            assert!(p99 < Duration::from_millis(5), "p99 {p99:?}");
        }
    }
}
