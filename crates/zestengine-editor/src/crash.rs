//! Crash diagnostics: a SIGSEGV/SIGBUS/SIGILL handler that prints a
//! symbolized Rust backtrace before the process dies, then re-raises the
//! signal with the default handler so a core dump is still produced.
//!
//! The systemd core dumps we captured only contained the null frame
//! (`ip 0`, a call through a null function pointer); this handler captures
//! the real Rust stack at the moment of the fault.

static PANICKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Install the crash handler (no-op on non-unix).
pub fn install() {
    // Remember a panic so a fault during the unwind is reported as fallout,
    // not as the root cause (the panic message above it is the real error).
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        PANICKED.store(true, std::sync::atomic::Ordering::Relaxed);
        previous(info);
    }));
    #[cfg(unix)]
    unsafe {
        for sig in [libc::SIGSEGV, libc::SIGBUS, libc::SIGILL, libc::SIGABRT] {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = handler as *const () as usize;
            // SA_RESETHAND restores the default handler after this delivery,
            // so the re-raise below produces the usual core dump.
            action.sa_flags = libc::SA_SIGINFO | libc::SA_RESETHAND;
            libc::sigemptyset(&mut action.sa_mask);
            libc::sigaction(sig, &action, std::ptr::null_mut());
        }
    }
}

#[cfg(unix)]
extern "C" fn handler(sig: i32, _info: *mut libc::siginfo_t, _ctx: *mut libc::c_void) {
    // force_capture ignores RUST_BACKTRACE and always unwinds. Allocating in a
    // signal handler isn't async-signal-safe, but we're already crashing, and a
    // best-effort backtrace beats the bare `ip 0` core.
    let bt = std::backtrace::Backtrace::force_capture();
    let name = match sig {
        libc::SIGSEGV => "SIGSEGV",
        libc::SIGBUS => "SIGBUS",
        libc::SIGILL => "SIGILL",
        libc::SIGABRT => "SIGABRT",
        _ => "signal",
    };
    let after = if PANICKED.load(std::sync::atomic::Ordering::Relaxed) {
        " while unwinding from an earlier panic (see the panic message above for the root cause)"
    } else {
        ""
    };
    eprintln!("\n==== zestengine crashed: {name} ({sig}){after} ====\n{bt}\n====");
    unsafe {
        libc::raise(sig);
    }
}
