//! `zestengine-build-server <project> [out]`: cook the Dedicated server target of a
//! project without the editor (`just build-server <project> [out]`). Writes
//! `<project>/build-server/` (binary, `server.ron`, `run.sh`, stripped `assets/`,
//! `build-report.txt`); `out` copies that folder somewhere else afterwards.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn copy_dir(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let dst = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &dst)?;
        } else {
            std::fs::copy(entry.path(), &dst)?;
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let mut opts = zestengine_engine::server_build::ServerBuildOptions::from_env();
    let mut positional = Vec::new();
    for a in std::env::args().skip(1) {
        match a.as_str() {
            "--strict-roles" => opts.strict_roles = true,
            "--pak" => opts.pak = true,
            _ => positional.push(a),
        }
    }
    let mut args = positional.into_iter();
    let Some(project) = args.next().filter(|a| !a.starts_with('-')) else {
        eprintln!(
            "usage: zestengine-build-server [--strict-roles] [--pak] <project-dir> [out-dir]\n  --strict-roles  cook warnings and reference lints are errors\n  --pak           also write assets.pak and point server.ron at it\nSet ZESTENGINE_SERVER_TARGET=<triple> to cross-compile."
        );
        return ExitCode::from(2);
    };
    let out = args.next().map(PathBuf::from);
    let project = match std::path::absolute(&project) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let built = zestengine_assets::load_project_file(&project)
        .map_err(|e| e.context("reading project.zestengine"))
        .and_then(|p| zestengine_engine::build_server_dir(&project, &p, opts));
    match built {
        Ok(exe) => {
            let folder = exe.parent().map(Path::to_path_buf).unwrap_or(project.join("build-server"));
            if let Some(out) = out {
                if let Err(e) = copy_dir(&folder, &out) {
                    eprintln!("error: copying to {}: {e}", out.display());
                    return ExitCode::FAILURE;
                }
                println!("server build copied to {}", out.display());
            } else {
                println!("server build in {}", folder.display());
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
