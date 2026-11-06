#!/bin/bash
set -euo pipefail

# Build a .deb package for ZestEngine
# Usage:
#   ./build-deb.sh                          build on this Debian/Ubuntu machine; on
#                                           other distros, same as --container
#   ./build-deb.sh --container [IMAGE]      build inside IMAGE (default ubuntu:24.04)
#                                           with podman or docker, from any distro
#
# The .deb only works on the release it was built on (or newer), so build one
# per target, e.g. --container ubuntu:24.04, --container ubuntu:25.04.
#
# Prerequisites for a native build (Debian/Ubuntu):
#   sudo apt install build-essential cmake python3 git pkg-config libasound2-dev dpkg-dev
#   plus Rust >= 1.85 from rustup (Ubuntu's packaged rustc is too old)
#   Network during the build: audionimbus downloads Steam Audio.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"

if [ "${1:-}" != "--container" ] && [ ! -f /etc/debian_version ]; then
    echo "Not a Debian/Ubuntu system: building in an ubuntu:24.04 container instead"
    set -- --container
fi

if [ "${1:-}" = "--container" ]; then
    IMAGE="${2:-ubuntu:24.04}"
    ENGINE=$(command -v podman || command -v docker || true)
    [ -n "$ENGINE" ] || { echo "podman or docker is required for --container" >&2; exit 1; }
    TAG="zestengine-deb-build:${IMAGE//[:\/]/-}"
    "$ENGINE" build --build-arg "BASE=$IMAGE" -t "$TAG" -f "$SCRIPT_DIR/Containerfile" "$SCRIPT_DIR"
    USER_ARGS=()
    [[ "$ENGINE" == *docker ]] && USER_ARGS=(--user "$(id -u):$(id -g)")
    "$ENGINE" run --rm "${USER_ARGS[@]}" \
        -v "$PROJECT_DIR:/src:Z" -w /src \
        -e CARGO_TARGET_DIR="/src/target/deb-${IMAGE//[:\/]/-}" \
        -e DEB_SUFFIX="${IMAGE//[:\/]/}" \
        "$TAG" pkg/deb/build-deb.sh
    exit 0
fi

VERSION=$(python3 tools/version.py)
ARCH=$(dpkg --print-architecture)
PKG_NAME="zestengine"
TARGET_DIR="${CARGO_TARGET_DIR:-target}"
DEB_VERSION="${VERSION}${DEB_SUFFIX:+-$DEB_SUFFIX}"
PKG_DIR="$SCRIPT_DIR/${PKG_NAME}_${DEB_VERSION}_${ARCH}"

echo "Building ZestEngine v${DEB_VERSION} .deb for ${ARCH}..."

cargo build --workspace --release --locked

release="$TARGET_DIR/release"
phonon="$release/libphonon.so"
if [[ ! -f "$phonon" ]]; then
    phonon=$(find "$release/build" -path '*/audionimbus-sys-*/out/lib/libphonon.so' -print -quit)
fi
for binary in zestengine zestengine-hub zestengine-relay; do
    test -f "$release/$binary"
done
if [[ ! -f "$phonon" ]]; then
    echo "Steam Audio libphonon.so is missing after the build." >&2
    exit 1
fi

rm -rf "$PKG_DIR"
libdir="$PKG_DIR/usr/lib/zestengine"
mkdir -p "$PKG_DIR/DEBIAN" "$PKG_DIR/usr/bin" "$libdir"
for binary in zestengine zestengine-hub zestengine-relay; do
    install -Dm755 "$release/$binary" "$libdir/$binary"
    strip "$libdir/$binary"
    ln -s "../lib/zestengine/$binary" "$PKG_DIR/usr/bin/$binary"
done
install -Dm755 "$phonon" "$libdir/libphonon.so"
install -Dm644 pkg/zestengine.desktop "$PKG_DIR/usr/share/applications/zestengine.desktop"
install -Dm644 pkg/zestengine-hub.desktop "$PKG_DIR/usr/share/applications/zestengine-hub.desktop"
install -Dm644 pkg/zestengine.svg "$PKG_DIR/usr/share/icons/hicolor/scalable/apps/zestengine.svg"
for size in 16 32 48 64 128 256 512; do
    install -Dm644 "pkg/icons/zestengine-${size}.png" \
        "$PKG_DIR/usr/share/icons/hicolor/${size}x${size}/apps/zestengine.png"
done
install -Dm644 LICENSE "$PKG_DIR/usr/share/doc/$PKG_NAME/copyright"
install -Dm644 NOTICE "$PKG_DIR/usr/share/doc/$PKG_NAME/NOTICE"
install -Dm644 crates/zestengine-engine/assets/licenses/steam-audio-LICENSE.txt \
    "$PKG_DIR/usr/share/doc/$PKG_NAME/steam-audio-LICENSE.txt"
install -Dm644 crates/zestengine-editor/assets/fonts/OFL.txt \
    "$PKG_DIR/usr/share/doc/$PKG_NAME/OFL.txt"
install -Dm644 README.md "$PKG_DIR/usr/share/doc/$PKG_NAME/README.md"

SHLIBS_DIR=$(mktemp -d)
trap 'rm -rf "$SHLIBS_DIR"' EXIT
mkdir -p "$SHLIBS_DIR/debian"
printf 'Source: %s\n\nPackage: %s\nArchitecture: any\n' "$PKG_NAME" "$PKG_NAME" > "$SHLIBS_DIR/debian/control"
SHLIBS_DEPENDS=$(cd "$SHLIBS_DIR" && dpkg-shlibdeps -O \
    "$libdir/zestengine" "$libdir/zestengine-hub" "$libdir/zestengine-relay" "$libdir/libphonon.so" \
    | sed -n 's/^shlibs:Depends=//p')

# Vulkan, X11 and Wayland are dlopen'd, so dpkg-shlibdeps cannot see them.
DLOPEN_DEPENDS="libvulkan1, libx11-6, libx11-xcb1, libxcursor1, libxi6, libxkbcommon0, libxkbcommon-x11-0, libwayland-client0, hicolor-icon-theme"
INSTALLED_SIZE=$(du -sk "$PKG_DIR" | cut -f1)

cat > "$PKG_DIR/DEBIAN/control" <<EOF
Package: zestengine
Version: ${DEB_VERSION}
Section: devel
Priority: optional
Architecture: ${ARCH}
Depends: ${SHLIBS_DEPENDS}, ${DLOPEN_DEPENDS}
Installed-Size: ${INSTALLED_SIZE}
Maintainer: Ryno Kotze <lemon.xah@gmail.com>
Homepage: https://github.com/lemonxah/zestengine
Description: 3D game engine and editor
 ZestEngine is a desktop game engine with a dockable editor, scenes,
 materials, and realtime global illumination. This package installs the
 editor, the Project Hub and the relay.
EOF

dpkg-deb --build --root-owner-group "$PKG_DIR"
rm -rf "$PKG_DIR"

echo ""
echo "Package built: ${PKG_DIR}.deb"
