//! Termination requests: SIGTERM / SIGINT on Unix, Ctrl+C / Ctrl+Break / console
//! close on Windows. The handler only sets a flag (async-signal-safe); the tick
//! loop polls [`requested`]. A second signal exits at once, for a stuck server.

use std::sync::atomic::{AtomicU32, Ordering};

static COUNT: AtomicU32 = AtomicU32::new(0);

/// A shutdown was requested.
pub fn requested() -> bool {
    COUNT.load(Ordering::Relaxed) > 0
}

/// Ask for a shutdown as a signal would (tests, `quit`).
pub fn request() {
    COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Forget earlier requests (tests).
pub fn reset() {
    COUNT.store(0, Ordering::Relaxed);
}

#[cfg(unix)]
extern "C" fn on_signal(_sig: libc::c_int) {
    if COUNT.fetch_add(1, Ordering::Relaxed) >= 1 {
        // SAFETY: _exit is async-signal-safe.
        unsafe { libc::_exit(130) };
    }
}

/// Install the handlers (once, at startup).
#[cfg(unix)]
pub fn install() -> std::io::Result<()> {
    for sig in [libc::SIGTERM, libc::SIGINT] {
        // SAFETY: the handler only touches an atomic and calls _exit.
        let prev = unsafe { libc::signal(sig, on_signal as extern "C" fn(libc::c_int) as libc::sighandler_t) };
        if prev == libc::SIG_ERR {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

#[cfg(windows)]
unsafe extern "system" fn on_ctrl(_kind: u32) -> i32 {
    if COUNT.fetch_add(1, Ordering::Relaxed) >= 1 {
        std::process::exit(130);
    }
    1
}

#[cfg(windows)]
pub fn install() -> std::io::Result<()> {
    use windows_sys::Win32::System::Console::SetConsoleCtrlHandler;
    // SAFETY: registers a handler that only touches an atomic.
    if unsafe { SetConsoleCtrlHandler(Some(on_ctrl), 1) } == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
pub fn install() -> std::io::Result<()> {
    Ok(())
}
