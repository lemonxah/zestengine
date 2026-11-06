set shell := ["bash", "-euo", "pipefail", "-c"]

# Show available commands.
default:
    @just --list

# Show the shared workspace version.
current:
    @python3 tools/version.py

# Bump the shared workspace version and refresh Cargo.lock (no dependency upgrades).
bump new_version:
    python3 tools/version.py {{quote(new_version)}}

# Run all workspace tests.
test:
    cargo test --workspace --locked

# Check every workspace target without producing executables, the player-only
# engine (`client` without the editor: what a built game links), and the GPU-less
# server build (`client` off: no renderer, XR, audio, gamepad, winit or egui),
# plus the tests that run without a renderer.
check:
    cargo check --workspace --all-targets --locked
    cargo check -p zestengine-engine --no-default-features --features client --locked
    cargo check -p zestengine-engine --no-default-features --locked
    cargo check -p zestengine-server --all-targets --locked
    cargo check -p zestengine --no-default-features --locked --examples
    ZESTENGINE_TEST_GPU=0 cargo test -p zestengine-engine --no-default-features --lib --locked

# Format Rust sources.
fmt:
    cargo fmt --all

# Build the workspace, including the editor, hub, relay and example plugin.
build:
    cargo build --workspace --release --locked

# Run the editor (build both executables so switching to the hub works).
[positional-arguments]
run *args:
    cargo build -p zestengine -p zestengine-hub --locked
    cargo run --bin zestengine --locked -- "$@"

# Build and run the dedicated server (no GPU) on a project: just server ~/games/arena --port 9000
[positional-arguments]
server project *args:
    cargo build -p zestengine-server --locked
    cargo run -p zestengine-server --locked -- --project "$1" "${@:2}"

# Build the release dedicated server binary (target/release/zestengine-server).
server-release:
    cargo build --release -p zestengine-server --locked

# Cook a project's Dedicated server target: stripped assets, server.ron, run.sh and the
# server binary (the project's own when it has components) in <project>/build-server,
# copied to [out] when given. ZESTENGINE_SERVER_TARGET=<triple> cross-compiles.
# just build-server ~/games/arena [~/deploy/arena]
[positional-arguments]
build-server project out="":
    cargo run --release -p zestengine-server --bin zestengine-build-server --locked -- "$1" ${2:+"$2"}

# Try `cargo check` of the dedicated server for cross targets (not part of CI; skips targets that are not installed).
check-server-cross *targets:
    scripts/check-server-cross.sh {{targets}}

# Open just the Project Hub (builds only the hub; it launches the editor binary that is
# already built next to it, so a broken editor build does not block the hub).
hub:
    cargo run --bin zestengine-hub --locked

# Open the Project Hub with its matching editor built first.
hub-full:
    cargo build -p zestengine -p zestengine-hub --locked
    cargo run --bin zestengine-hub --locked

# Build and install on Linux; override with: just install "$HOME/.local" ""
install prefix="/usr" sudo="sudo": build
    bash tools/install.sh {{quote(prefix)}} {{quote(sudo)}}

# Bump, commit, tag, push and create a GitHub release (requires a clean checkout).
release new_version msg="":
    #!/usr/bin/env bash
    set -euo pipefail
    version={{quote(new_version)}}
    message={{quote(msg)}}
    command -v gh >/dev/null
    gh auth status
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "Commit or stash your changes before releasing." >&2
        exit 1
    fi
    branch=$(git symbolic-ref --short HEAD)
    git rev-parse --verify '@{upstream}' >/dev/null
    if git show-ref --verify --quiet "refs/tags/v$version"; then
        echo "Tag v$version already exists." >&2
        exit 1
    fi
    just bump "$version"
    title="v$version"
    if [[ -n "$message" ]]; then title+=": $message"; fi
    git add -- Cargo.toml Cargo.lock
    git commit -m "$title" -- Cargo.toml Cargo.lock
    git tag -a "v$version" -m "$title"
    git push --atomic origin "$branch" "refs/tags/v$version"
    gh release create "v$version" --verify-tag --title "$title" --generate-notes

# ---- Packages ---------------------------------------------------------------
# The .deb works on the Ubuntu release it was built for and newer.
# The .rpm works on that Fedora release only.

deb image="ubuntu:24.04":
    pkg/deb/build-deb.sh --container {{image}}

rpm image="fedora:latest":
    pkg/rpm/build-rpm.sh --container {{image}}

# Build with Nix. The result is symlinked at ./result.
nix:
    nix build ./pkg/nix
