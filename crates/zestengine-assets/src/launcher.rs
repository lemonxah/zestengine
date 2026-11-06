//! Finding and starting the sibling zestengine executables: the Project Hub starts
//! the editor (`zestengine <project-dir>`, `zestengine --new-project ...`) and the
//! editor starts the hub. Both live in the same install / target folder.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

/// The editor binary's file stem.
pub const EDITOR_BIN: &str = "zestengine";
/// The Project Hub binary's file stem.
pub const HUB_BIN: &str = "zestengine-hub";
/// Overrides where the hub looks for the editor.
pub const EDITOR_ENV: &str = "ZESTENGINE_EDITOR";
/// Overrides where the editor looks for the hub.
pub const HUB_ENV: &str = "ZESTENGINE_HUB";

/// `stem` plus the platform's executable suffix (`.exe` on Windows).
pub fn exe_name(stem: &str) -> String {
    format!("{stem}{}", std::env::consts::EXE_SUFFIX)
}

/// Places to look for the `stem` executable, in priority order: an explicit
/// path (a setting), the environment override, next to `exe_dir` (the running
/// executable's folder), then, for a cargo `target/debug` or `target/release`
/// folder, the other profile's folder.
pub fn candidates(stem: &str, explicit: Option<&Path>, env_value: Option<&OsStr>, exe_dir: Option<&Path>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = explicit.filter(|p| !p.as_os_str().is_empty()) {
        out.push(p.to_path_buf());
    }
    if let Some(v) = env_value.filter(|v| !v.is_empty()) {
        out.push(PathBuf::from(v));
    }
    if let Some(dir) = exe_dir {
        let name = exe_name(stem);
        out.push(dir.join(&name));
        let profile = dir.file_name().and_then(OsStr::to_str);
        let other = match profile {
            Some("debug") => Some("release"),
            Some("release") => Some("debug"),
            _ => None,
        };
        if let (Some(other), Some(parent)) = (other, dir.parent()) {
            out.push(parent.join(other).join(&name));
        }
    }
    out
}

/// The first candidate that exists (see [`candidates`]), using the running
/// executable's folder and `env_var`. The error lists every place searched.
pub fn find_executable(stem: &str, explicit: Option<&Path>, env_var: &str) -> Result<PathBuf> {
    let env_value = std::env::var_os(env_var);
    let exe = std::env::current_exe().ok();
    let exe_dir = exe.as_deref().and_then(Path::parent);
    find_in(stem, &candidates(stem, explicit, env_value.as_deref(), exe_dir), env_var)
}

fn find_in(stem: &str, candidates: &[PathBuf], env_var: &str) -> Result<PathBuf> {
    if let Some(found) = candidates.iter().find(|p| p.is_file()) {
        return Ok(found.clone());
    }
    let searched: Vec<String> = candidates.iter().map(|p| p.display().to_string()).collect();
    bail!(
        "could not find the `{}` executable (searched: {}). Build it (cargo build -p {stem}) or set {env_var}.",
        exe_name(stem),
        if searched.is_empty() { "nowhere".to_string() } else { searched.join(", ") }
    )
}

/// A process to start: program, arguments and working folder. Arguments are
/// passed as-is (never through a shell), so paths with spaces stay one argument.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchCommand {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub cwd: Option<PathBuf>,
}

impl LaunchCommand {
    /// `zestengine <project>`, run from inside the project folder.
    pub fn open_project(editor: &Path, project: &Path) -> Self {
        Self { program: editor.to_path_buf(), args: vec![project.as_os_str().to_owned()], cwd: Some(project.to_path_buf()) }
    }

    /// `zestengine --new-project <parent> <name> <template>` (headless scaffold).
    pub fn new_project(editor: &Path, parent: &Path, name: &str, template: &str) -> Self {
        Self {
            program: editor.to_path_buf(),
            args: vec!["--new-project".into(), parent.as_os_str().to_owned(), name.into(), template.into()],
            cwd: None,
        }
    }

    /// The hub, with no arguments.
    pub fn hub(hub: &Path) -> Self {
        Self { program: hub.to_path_buf(), args: Vec::new(), cwd: None }
    }

    pub fn command(&self) -> std::process::Command {
        let mut c = std::process::Command::new(&self.program);
        c.args(&self.args);
        if let Some(dir) = &self.cwd {
            c.current_dir(dir);
        }
        c
    }

    /// The command line as a shell would need it, for logs and error messages.
    pub fn display(&self) -> String {
        std::iter::once(self.program.as_os_str())
            .chain(self.args.iter().map(OsString::as_os_str))
            .map(|a| shell_quote(&a.to_string_lossy()))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Quote one argument for display: unchanged when it has no special
/// characters, else single quotes (POSIX) or double quotes (Windows).
pub fn shell_quote(arg: &str) -> String {
    let plain = !arg.is_empty()
        && arg.chars().all(|c| c.is_alphanumeric() || matches!(c, '/' | '\\' | '.' | '-' | '_' | ':' | '=' | '+' | ','));
    if plain {
        arg.to_string()
    } else if cfg!(windows) {
        format!("\"{}\"", arg.replace('"', "\\\""))
    } else {
        format!("'{}'", arg.replace('\'', "'\\''"))
    }
}

/// Open `dir` in the system file manager.
pub fn reveal_in_file_manager(dir: &Path) -> std::io::Result<()> {
    let program = if cfg!(windows) {
        "explorer"
    } else if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    std::process::Command::new(program).arg(dir).spawn().map(|_| ())
}

/// Host OS family for [`show_in_file_manager_argv`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileManagerOs {
    Linux,
    Windows,
    MacOs,
}

impl FileManagerOs {
    pub fn current() -> Self {
        if cfg!(windows) {
            Self::Windows
        } else if cfg!(target_os = "macos") {
            Self::MacOs
        } else {
            Self::Linux
        }
    }

    /// Context-menu label for "show this item in the OS file manager".
    pub fn menu_label(self) -> &'static str {
        match self {
            Self::Linux => "Show in File Browser",
            Self::Windows => "Show in Explorer",
            Self::MacOs => "Reveal in Finder",
        }
    }
}

/// `file://` URI with percent-encoding (bytes outside the unreserved set and `/`).
pub fn file_uri(path: &Path) -> String {
    let s = path.to_string_lossy();
    let mut out = String::from("file://");
    for b in s.as_bytes() {
        match *b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => out.push(*b as char),
            b => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Argument vectors (program first), tried in order until one launches, that show
/// `path` in the file manager with the item selected where the platform supports it.
/// A folder is opened itself on Linux (xdg-open fallback); no shell is involved.
pub fn show_in_file_manager_argv(os: FileManagerOs, path: &Path) -> Vec<Vec<std::ffi::OsString>> {
    use std::ffi::OsString;
    let o = |s: &str| OsString::from(s);
    match os {
        FileManagerOs::Windows => {
            let mut sel = OsString::from("/select,");
            sel.push(path);
            vec![vec![o("explorer"), sel]]
        }
        FileManagerOs::MacOs => vec![vec![o("open"), o("-R"), path.into()]],
        FileManagerOs::Linux => {
            let open_target = if path.is_dir() { path } else { path.parent().unwrap_or(path) };
            vec![
                vec![
                    o("dbus-send"),
                    o("--session"),
                    o("--print-reply"),
                    o("--reply-timeout=1000"),
                    o("--dest=org.freedesktop.FileManager1"),
                    o("--type=method_call"),
                    o("/org/freedesktop/FileManager1"),
                    o("org.freedesktop.FileManager1.ShowItems"),
                    OsString::from(format!("array:string:{}", file_uri(path))),
                    o("string:"),
                ],
                vec![o("xdg-open"), open_target.into()],
            ]
        }
    }
}

/// Show `path` (file or folder) in the system file manager, selected where possible.
/// The file manager itself is spawned detached; only the D-Bus call is awaited (it
/// replies at once, 1 s timeout) so a missing service falls through to xdg-open.
pub fn show_in_file_manager(path: &Path) -> std::io::Result<()> {
    if !path.exists() {
        return Err(std::io::Error::new(std::io::ErrorKind::NotFound, format!("{} does not exist", path.display())));
    }
    let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let os = FileManagerOs::current();
    let mut last = None;
    for argv in show_in_file_manager_argv(os, &path) {
        let mut cmd = std::process::Command::new(&argv[0]);
        cmd.args(&argv[1..]).stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
        let r = if os == FileManagerOs::Linux && argv[0] == "dbus-send" {
            // Short-lived: wait so a missing FileManager1 service reports failure.
            cmd.status().and_then(|s| if s.success() { Ok(()) } else { Err(std::io::Error::other("FileManager1 unavailable")) })
        } else {
            cmd.spawn().map(|mut c| {
                // Reap in the background so no zombie is left; never block the caller.
                std::thread::spawn(move || {
                    let _ = c.wait();
                });
            })
        };
        match r {
            Ok(()) => return Ok(()),
            Err(e) => last = Some(e),
        }
    }
    Err(last.unwrap_or_else(|| std::io::Error::other("no file manager command")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn show_in_file_manager_commands() {
        let p = Path::new("/home/u/my proj/fbx file.fbx");
        let w = show_in_file_manager_argv(FileManagerOs::Windows, Path::new("C:\\a b\\x.fbx"));
        assert_eq!(w, vec![vec!["explorer".into(), std::ffi::OsString::from("/select,C:\\a b\\x.fbx")]]);
        let m = show_in_file_manager_argv(FileManagerOs::MacOs, p);
        assert_eq!(m, vec![vec![std::ffi::OsString::from("open"), "-R".into(), p.into()]]);
        let l = show_in_file_manager_argv(FileManagerOs::Linux, p);
        assert_eq!(l.len(), 2);
        assert_eq!(l[0][0], "dbus-send");
        assert_eq!(l[0][8], "array:string:file:///home/u/my%20proj/fbx%20file.fbx");
        assert_eq!(l[1], vec![std::ffi::OsString::from("xdg-open"), "/home/u/my proj".into()]);
        assert_eq!(file_uri(Path::new("/a/é")), "file:///a/%C3%A9");
        assert_eq!(FileManagerOs::MacOs.menu_label(), "Reveal in Finder");
        assert_eq!(FileManagerOs::Windows.menu_label(), "Show in Explorer");
        assert!(show_in_file_manager(Path::new("/definitely/not/here/x")).is_err());
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("zestengine-launcher-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn candidate_order_and_profile_fallback() {
        let dir = Path::new("/opt/x/target/debug");
        let c = candidates("zestengine", Some(Path::new("/custom/zestengine")), Some(OsStr::new("/env/zestengine")), Some(dir));
        assert_eq!(
            c,
            [
                PathBuf::from("/custom/zestengine"),
                PathBuf::from("/env/zestengine"),
                dir.join(exe_name("zestengine")),
                PathBuf::from("/opt/x/target/release").join(exe_name("zestengine")),
            ]
        );
        // Empty setting / env values are ignored; an install folder has no profile fallback.
        let c = candidates("zestengine", Some(Path::new("")), Some(OsStr::new("")), Some(Path::new("/usr/bin")));
        assert_eq!(c, [Path::new("/usr/bin").join(exe_name("zestengine"))]);
    }

    #[test]
    fn finds_the_first_existing_candidate() {
        let dir = temp_dir("find");
        let release = dir.join("target/release");
        let debug = dir.join("target/debug");
        std::fs::create_dir_all(&release).unwrap();
        std::fs::create_dir_all(&debug).unwrap();
        // Only the release editor exists: a debug hub falls back to it.
        let editor = release.join(exe_name(EDITOR_BIN));
        std::fs::write(&editor, b"").unwrap();
        let c = candidates(EDITOR_BIN, None, None, Some(&debug));
        assert_eq!(find_in(EDITOR_BIN, &c, EDITOR_ENV).unwrap(), editor);
        // A sibling in the hub's own folder wins over the other profile.
        let sibling = debug.join(exe_name(EDITOR_BIN));
        std::fs::write(&sibling, b"").unwrap();
        assert_eq!(find_in(EDITOR_BIN, &c, EDITOR_ENV).unwrap(), sibling);
        // An explicit path that exists wins over both.
        let custom = dir.join("my editor");
        std::fs::write(&custom, b"").unwrap();
        let c = candidates(EDITOR_BIN, Some(&custom), None, Some(&debug));
        assert_eq!(find_in(EDITOR_BIN, &c, EDITOR_ENV).unwrap(), custom);
        // Nothing found: the error names what was searched and the override.
        let c = candidates(HUB_BIN, None, None, Some(&dir.join("nowhere")));
        let err = format!("{:#}", find_in(HUB_BIN, &c, HUB_ENV).unwrap_err());
        assert!(err.contains(HUB_ENV) && err.contains("nowhere"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn launch_commands_keep_paths_with_spaces_whole() {
        let editor = Path::new("/opt/zestengine engine/zestengine");
        let project = Path::new("/home/me/My Games/space shooter");
        let open = LaunchCommand::open_project(editor, project);
        assert_eq!(open.args, [OsString::from("/home/me/My Games/space shooter")]);
        assert_eq!(open.cwd.as_deref(), Some(project));
        let cmd = open.command();
        assert_eq!(cmd.get_program(), editor.as_os_str());
        assert_eq!(cmd.get_args().collect::<Vec<_>>(), [project.as_os_str()]);
        assert_eq!(cmd.get_current_dir(), Some(project));

        let new = LaunchCommand::new_project(editor, Path::new("/home/me/My Games"), "it's mine", "fps");
        let args: Vec<&OsStr> = new.args.iter().map(OsString::as_os_str).collect();
        assert_eq!(args, ["--new-project", "/home/me/My Games", "it's mine", "fps"]);
        if !cfg!(windows) {
            assert_eq!(
                new.display(),
                r#"'/opt/zestengine engine/zestengine' --new-project '/home/me/My Games' 'it'\''s mine' fps"#
            );
        }
        assert_eq!(shell_quote("plain/path-1.0"), "plain/path-1.0");
        assert_ne!(shell_quote(""), "");
    }
}
