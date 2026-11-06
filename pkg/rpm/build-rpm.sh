#!/bin/bash
set -euo pipefail

# Build an .rpm package for ZestEngine
# Usage:
#   ./build-rpm.sh                          build on this Fedora machine; on
#                                           other distros, same as --container
#   ./build-rpm.sh --container [IMAGE]      build inside IMAGE (default fedora:latest)
#                                           with podman or docker, from any distro
#
# The .rpm only works on the Fedora release it was built for, so build one per
# target, e.g. --container fedora:42, --container fedora:43.
#
# Prerequisites for a native build (Fedora):
#   sudo dnf install rpm-build 'dnf-command(builddep)'
#   sudo dnf builddep pkg/rpm/zestengine.spec
#   Network during the build: audionimbus downloads Steam Audio.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"

if [ "${1:-}" != "--container" ] && [ ! -f /etc/fedora-release ]; then
    echo "Not a Fedora system: building in a fedora:latest container instead"
    set -- --container
fi

if [ "${1:-}" = "--container" ]; then
    IMAGE="${2:-fedora:latest}"
    ENGINE=$(command -v podman || command -v docker || true)
    [ -n "$ENGINE" ] || { echo "podman or docker is required for --container" >&2; exit 1; }
    TAG="zestengine-rpm-build:${IMAGE//[:\/]/-}"
    PULL_ARG=--pull=newer
    [[ "$ENGINE" == *docker ]] && PULL_ARG=--pull
    "$ENGINE" build "$PULL_ARG" --build-arg "BASE=$IMAGE" -t "$TAG" -f "$SCRIPT_DIR/Containerfile" "$SCRIPT_DIR"
    USER_ARGS=()
    [[ "$ENGINE" == *docker ]] && USER_ARGS=(--user "$(id -u):$(id -g)" -e HOME=/tmp)
    "$ENGINE" run --rm "${USER_ARGS[@]}" \
        -v "$PROJECT_DIR:/src:Z" -w /src \
        "$TAG" pkg/rpm/build-rpm.sh
    exit 0
fi

VERSION=$(python3 tools/version.py)
echo "Building ZestEngine v${VERSION} .rpm..."

RPMBUILD_DIR="$SCRIPT_DIR/rpmbuild"
mkdir -p "$RPMBUILD_DIR"/{BUILD,RPMS,SOURCES,SPECS,SRPMS}

TARBALL_DIR="zestengine-${VERSION}"
tar -czf "$RPMBUILD_DIR/SOURCES/zestengine-${VERSION}.tar.gz" \
    --exclude-vcs \
    --exclude=./target \
    --exclude=./target-win \
    --exclude=./target-android \
    --exclude=./example-game \
    --exclude=./extracted \
    --exclude=./textures \
    --exclude=./models \
    --exclude=./scenes \
    --exclude=./animations \
    --exclude=./pkg/rpm/rpmbuild \
    --exclude='./pkg/deb/zestengine_*' \
    --transform "s,^\.,${TARBALL_DIR}," \
    .

sed "s/^Version:.*/Version:        ${VERSION}/" "$SCRIPT_DIR/zestengine.spec" > "$RPMBUILD_DIR/SPECS/zestengine.spec"

rpmbuild --define "_topdir $RPMBUILD_DIR" -bb "$RPMBUILD_DIR/SPECS/zestengine.spec"

echo ""
echo "RPM built in: $RPMBUILD_DIR/RPMS/"
find "$RPMBUILD_DIR/RPMS/" -name "*.rpm" -print
