Name:           zestengine
Version:        0.1.0
Release:        1%{?dist}
Summary:        3D game engine and editor

License:        LicenseRef-ZestEngine
URL:            https://github.com/lemonxah/zestengine
Source0:        %{name}-%{version}.tar.gz

# Edition 2024 needs Rust >= 1.85. astcenc compiles as C++.
# audionimbus downloads Steam Audio during the build.
BuildRequires:  rust >= 1.85
BuildRequires:  cargo
BuildRequires:  gcc-c++
BuildRequires:  make
BuildRequires:  cmake
BuildRequires:  python3
BuildRequires:  git
BuildRequires:  pkgconfig
BuildRequires:  alsa-lib-devel
BuildRequires:  curl

Requires:       vulkan-loader
Requires:       libX11
Requires:       libX11-xcb
Requires:       libXcursor
Requires:       libXi
Requires:       libxkbcommon
Requires:       libxkbcommon-x11
Requires:       wayland
Requires:       alsa-lib
Requires:       hicolor-icon-theme

%description
ZestEngine is a desktop game engine with a dockable editor, scenes, materials,
and realtime global illumination. This package installs the editor, the
Project Hub and the relay.

%prep
%autosetup

%build
export CARGO_TARGET_DIR=target
cargo build --workspace --release --locked

%install
release=target/release
phonon="$release/libphonon.so"
if [ ! -f "$phonon" ]; then
    phonon=$(find "$release/build" -path '*/audionimbus-sys-*/out/lib/libphonon.so' -print -quit)
fi
libdir=%{buildroot}/usr/lib/zestengine
mkdir -p "$libdir" %{buildroot}%{_bindir}
for binary in zestengine zestengine-hub zestengine-relay; do
    install -Dm755 "$release/$binary" "$libdir/$binary"
    ln -s "../lib/zestengine/$binary" "%{buildroot}%{_bindir}/$binary"
done
install -Dm755 "$phonon" "$libdir/libphonon.so"
install -Dm644 pkg/zestengine.desktop %{buildroot}%{_datadir}/applications/zestengine.desktop
install -Dm644 pkg/zestengine-hub.desktop %{buildroot}%{_datadir}/applications/zestengine-hub.desktop
install -Dm644 pkg/zestengine.svg %{buildroot}%{_datadir}/icons/hicolor/scalable/apps/zestengine.svg
for size in 16 32 48 64 128 256 512; do
    install -Dm644 "pkg/icons/zestengine-${size}.png" \
        "%{buildroot}%{_datadir}/icons/hicolor/${size}x${size}/apps/zestengine.png"
done

%files
%license LICENSE
%license NOTICE
%license crates/zestengine-engine/assets/licenses/steam-audio-LICENSE.txt
%license crates/zestengine-editor/assets/fonts/OFL.txt
%doc README.md
/usr/lib/zestengine/zestengine
/usr/lib/zestengine/zestengine-hub
/usr/lib/zestengine/zestengine-relay
/usr/lib/zestengine/libphonon.so
%{_bindir}/zestengine
%{_bindir}/zestengine-hub
%{_bindir}/zestengine-relay
%{_datadir}/applications/zestengine.desktop
%{_datadir}/applications/zestengine-hub.desktop
%{_datadir}/icons/hicolor/scalable/apps/zestengine.svg
%{_datadir}/icons/hicolor/16x16/apps/zestengine.png
%{_datadir}/icons/hicolor/32x32/apps/zestengine.png
%{_datadir}/icons/hicolor/48x48/apps/zestengine.png
%{_datadir}/icons/hicolor/64x64/apps/zestengine.png
%{_datadir}/icons/hicolor/128x128/apps/zestengine.png
%{_datadir}/icons/hicolor/256x256/apps/zestengine.png
%{_datadir}/icons/hicolor/512x512/apps/zestengine.png

%changelog
* Sun Nov 01 2026 Ryno Kotze <lemon.xah@gmail.com> - 0.1.0-1
- Initial package
