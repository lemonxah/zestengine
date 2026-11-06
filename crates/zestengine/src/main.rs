use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};
use zestengine_assets::PROJECT_FILE_NAME;
use zestengine_assets::launcher::{HUB_BIN, HUB_ENV, LaunchCommand, find_executable};
use zestengine_assets::recent_projects::RecentProjects;

const USAGE: &str = "usage:
  zestengine <project-dir>                 open a project in the editor
  zestengine <file.scene | model>          open a scene or model (project = the folder holding project.zestengine)
  zestengine                               open the project in the current folder, else start the Project Hub
  zestengine --hub                         start the Project Hub
  zestengine --new-project <parent-dir> <name> [template]
  zestengine --build [project-dir] [--target host|windows|android|quest]";

/// What the command line asks for.
#[derive(Debug, PartialEq)]
enum Launch {
    /// No arguments: see [`resolve_default`].
    Default,
    Hub,
    Help,
    /// Open this project folder.
    Project(PathBuf),
    /// Open a scene or model file.
    File(PathBuf),
    NewProject { parent: PathBuf, name: String, template: String },
    Build { dir: Option<PathBuf>, target: String },
}

fn parse_args(args: &[String]) -> Result<Launch> {
    let Some(first) = args.get(1) else {
        return Ok(Launch::Default);
    };
    match first.as_str() {
        "--hub" => Ok(Launch::Hub),
        "-h" | "--help" => Ok(Launch::Help),
        "--new-project" => {
            let (Some(parent), Some(name)) = (args.get(2), args.get(3)) else {
                let ids: Vec<&str> = zestengine_engine::templates::TEMPLATES.iter().map(|t| t.id).collect();
                bail!("usage: zestengine --new-project <parent-dir> <name> [template: {}]", ids.join(" | "));
            };
            let template = args.get(4).map_or(zestengine_engine::templates::DEFAULT_TEMPLATE, String::as_str);
            Ok(Launch::NewProject { parent: PathBuf::from(parent), name: name.clone(), template: template.to_string() })
        }
        "--build" => {
            let mut dir = None;
            let mut target = "host".to_string();
            let mut rest = args[2..].iter();
            while let Some(arg) = rest.next() {
                if arg == "--target" {
                    target = rest.next().cloned().unwrap_or_default();
                } else {
                    dir = Some(PathBuf::from(arg));
                }
            }
            Ok(Launch::Build { dir, target })
        }
        flag if flag.starts_with("--") => bail!("unknown option {flag}\n{USAGE}"),
        path => {
            let path = PathBuf::from(path);
            if path.is_dir() {
                Ok(Launch::Project(path))
            } else if path.is_file() {
                Ok(Launch::File(path))
            } else {
                bail!("{} does not exist\n{USAGE}", path.display())
            }
        }
    }
}

/// What a bare `zestengine` does.
#[derive(Debug, PartialEq)]
enum DefaultAction {
    /// The current folder is a project: open it (the long-standing behaviour).
    OpenCurrent,
    LaunchHub(PathBuf),
    /// No hub found: reopen the last project.
    OpenLast(PathBuf),
    Usage,
}

fn resolve_default(cwd: &Path, hub: Option<PathBuf>, last: Option<PathBuf>) -> DefaultAction {
    if cwd.join(PROJECT_FILE_NAME).is_file() {
        DefaultAction::OpenCurrent
    } else if let Some(hub) = hub {
        DefaultAction::LaunchHub(hub)
    } else if let Some(last) = last.filter(|p| p.join(PROJECT_FILE_NAME).is_file()) {
        DefaultAction::OpenLast(last)
    } else {
        DefaultAction::Usage
    }
}

/// The folder at or above `start` holding project.zestengine.
fn project_root_of(start: &Path) -> Option<PathBuf> {
    start.ancestors().find(|d| d.join(PROJECT_FILE_NAME).is_file()).map(Path::to_path_buf)
}

fn find_hub() -> Result<PathBuf> {
    find_executable(HUB_BIN, None, HUB_ENV)
}

fn launch_hub(hub: &Path) -> Result<()> {
    let cmd = LaunchCommand::hub(hub);
    cmd.command().spawn().with_context(|| format!("starting the Project Hub ({})", cmd.display()))?;
    Ok(())
}

/// Run the editor on `root` (the editor treats the working folder as the project).
#[cfg(feature = "editor")]
fn run_editor(root: &Path, scene_path: Option<String>) -> Result<()> {
    std::env::set_current_dir(root).with_context(|| format!("entering {}", root.display()))?;
    let config = zestengine_editor::AppConfig { scene_path, ..Default::default() };
    zestengine_editor::run(config)
}

/// A build without the editor feature (a GPU-less server build) cannot open projects.
#[cfg(not(feature = "editor"))]
fn run_editor(_root: &Path, _scene_path: Option<String>) -> Result<()> {
    bail!("this build has no editor (built without the `editor` feature); run the headless examples or your game's server binary")
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    zestengine_engine::init_logging();
    match parse_args(&args)? {
        Launch::Help => {
            println!("{USAGE}");
            Ok(())
        }
        Launch::NewProject { parent, name, template } => {
            let root = zestengine_engine::scaffold_project(&parent, &name, &template)?;
            println!("created project at {}", root.display());
            Ok(())
        }
        Launch::Build { dir, target } => {
            let target = zestengine_engine::BuildTarget::parse(&target)
                .ok_or_else(|| anyhow::anyhow!("unknown build target '{target}' (host | windows | android | quest | server)"))?;
            let dir = match dir {
                Some(d) => d,
                None => std::env::current_dir()?,
            };
            let exe = zestengine_engine::build_project_dir(&dir, target)?;
            println!("built {}", exe.display());
            Ok(())
        }
        Launch::Hub => launch_hub(&find_hub()?),
        Launch::Project(dir) => run_editor(&dir, None),
        Launch::File(file) => {
            let file = std::path::absolute(&file)?;
            let root = file.parent().and_then(project_root_of).map_or_else(std::env::current_dir, Ok)?;
            run_editor(&root, Some(file.to_string_lossy().into_owned()))
        }
        Launch::Default => {
            let cwd = std::env::current_dir()?;
            let last = RecentProjects::default_path().and_then(|p| RecentProjects::load(&p).last_opened);
            match resolve_default(&cwd, find_hub().ok(), last) {
                DefaultAction::OpenCurrent => run_editor(&cwd, None),
                DefaultAction::LaunchHub(hub) => launch_hub(&hub),
                DefaultAction::OpenLast(root) => run_editor(&root, None),
                DefaultAction::Usage => bail!("no project here and the Project Hub was not found\n{USAGE}"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("zestengine-cli-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn args(list: &[&str]) -> Vec<String> {
        std::iter::once("zestengine").chain(list.iter().copied()).map(String::from).collect()
    }

    #[test]
    fn hub_and_direct_open_arguments() {
        let dir = temp_dir("args");
        let project = dir.join("My Game");
        std::fs::create_dir_all(&project).unwrap();
        let scene = project.join("main.scene");
        std::fs::write(&scene, "").unwrap();

        assert_eq!(parse_args(&args(&[])).unwrap(), Launch::Default);
        assert_eq!(parse_args(&args(&["--hub"])).unwrap(), Launch::Hub);
        assert_eq!(parse_args(&args(&["--help"])).unwrap(), Launch::Help);
        assert_eq!(parse_args(&args(&[project.to_str().unwrap()])).unwrap(), Launch::Project(project.clone()));
        assert_eq!(parse_args(&args(&[scene.to_str().unwrap()])).unwrap(), Launch::File(scene.clone()));
        assert!(parse_args(&args(&[dir.join("missing").to_str().unwrap()])).is_err());
        assert!(parse_args(&args(&["--bogus"])).is_err());
        assert_eq!(
            parse_args(&args(&["--new-project", "/tmp/x y", "game"])).unwrap(),
            Launch::NewProject { parent: PathBuf::from("/tmp/x y"), name: "game".into(), template: "empty".into() }
        );
        assert!(parse_args(&args(&["--new-project", "/tmp"])).is_err());
        assert_eq!(
            parse_args(&args(&["--build", "--target", "windows", "proj"])).unwrap(),
            Launch::Build { dir: Some(PathBuf::from("proj")), target: "windows".into() }
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bare_zestengine_prefers_current_project_then_hub_then_last() {
        let dir = temp_dir("default");
        let empty = dir.join("empty");
        let project = dir.join("proj");
        std::fs::create_dir_all(&empty).unwrap();
        std::fs::create_dir_all(project.join("scenes")).unwrap();
        std::fs::write(project.join("project.zestengine"), "()").unwrap();
        let hub = PathBuf::from("/opt/zestengine/zestengine-hub");

        assert_eq!(resolve_default(&project, Some(hub.clone()), None), DefaultAction::OpenCurrent);
        assert_eq!(resolve_default(&empty, Some(hub.clone()), Some(project.clone())), DefaultAction::LaunchHub(hub));
        assert_eq!(resolve_default(&empty, None, Some(project.clone())), DefaultAction::OpenLast(project.clone()));
        assert_eq!(resolve_default(&empty, None, Some(dir.join("gone"))), DefaultAction::Usage);
        assert_eq!(resolve_default(&empty, None, None), DefaultAction::Usage);
        assert_eq!(project_root_of(&project.join("scenes")), Some(project.clone()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The hub creates projects through `zestengine --new-project`; every template
    /// must give a project whose project.zestengine and boot scene load.
    #[test]
    fn new_project_from_each_template_is_loadable() {
        let parent = temp_dir("new");
        for t in zestengine_engine::templates::TEMPLATES {
            let name = format!("{} game", t.id);
            let launch = parse_args(&args(&["--new-project", parent.to_str().unwrap(), &name, t.id])).unwrap();
            let Launch::NewProject { parent, name, template } = launch else { panic!("{launch:?}") };
            let root = zestengine_engine::scaffold_project(&parent, &name, &template).unwrap();
            let project = zestengine_assets::load_project_file(&root).unwrap();
            let boot = project.boot_scene.expect("boot scene");
            let scene = zestengine_assets::load_scene_file(root.join(&boot)).unwrap();
            assert!(!scene.entries.is_empty(), "{}: boot scene has objects", t.id);
        }
        let _ = std::fs::remove_dir_all(&parent);
    }
}
