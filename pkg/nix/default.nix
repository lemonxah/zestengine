{
  lib,
  rustPlatform,
  pkgs,
  cmake,
  python3,
  git,
  pkg-config,
  alsa-lib,
  vulkan-loader,
  libxkbcommon,
  wayland,
  xorg,
}:

let
  root = ../..;
  # audionimbus-sys 4.8.2 downloads Steam Audio 4.8.1 with curl during the
  # build. The Nix sandbox has no network, so curl for that one URL is served
  # from the store.
  steamaudio = pkgs.fetchurl {
    url = "https://github.com/ValveSoftware/steam-audio/releases/download/v4.8.1/steamaudio_4.8.1.zip";
    hash = "sha256-Sgql7BF2848LCZOjfCJZ2ehvJ+ItXiT4PsTDy5odVEk=";
  };
  steamCurl = pkgs.writeShellScriptBin "curl" ''
    out=""
    prev=""
    url=""
    for arg in "$@"; do
      if [ "$prev" = "-o" ]; then
        out=$arg
      fi
      case "$arg" in
        -*) ;;
        *) url=$arg ;;
      esac
      prev=$arg
    done
    case "$url" in
      *steamaudio_4.8.1.zip*)
        cp -f ${steamaudio} "$out"
        exit 0
        ;;
    esac
    exec ${pkgs.curl}/bin/curl "$@"
  '';
  drop = [
    "target"
    "target-win"
    "target-android"
    "example-game"
    "extracted"
    "textures"
    "models"
    "scenes"
    "animations"
  ];
  runtimeLibs = [
    vulkan-loader
    libxkbcommon
    wayland
    xorg.libX11
    xorg.libXcursor
    xorg.libXi
    alsa-lib
  ];
in
rustPlatform.buildRustPackage {
  pname = "zestengine";
  version = (lib.importTOML (root + "/Cargo.toml")).workspace.package.version;

  src = lib.cleanSourceWith {
    src = lib.cleanSource root;
    filter = path: _type:
      let base = baseNameOf (toString path);
      in !(builtins.elem base drop);
  };

  cargoLock.lockFile = root + "/Cargo.lock";

  cargoBuildFlags = [ "--workspace" ];

  # The root Cargo.toml is a virtual workspace, so there is nothing for
  # `cargo install` to install. cmake is only here for native build scripts.
  dontUseCmakeConfigure = true;
  installPhase = ''
    runHook preInstall
    runHook postInstall
  '';

  nativeBuildInputs = [
    cmake
    python3
    git
    pkg-config
    steamCurl
  ];

  buildInputs = runtimeLibs;

  postInstall = ''
    libdir=$out/lib/zestengine
    mkdir -p "$libdir" $out/bin
    for binary in zestengine zestengine-hub zestengine-relay; do
      install -Dm755 "target/release/$binary" "$libdir/$binary"
      ln -s "../lib/zestengine/$binary" "$out/bin/$binary"
    done
    phonon=target/release/libphonon.so
    if [ ! -f "$phonon" ]; then
      phonon=$(find target -path '*/audionimbus-sys-*/out/lib/libphonon.so' -print -quit)
    fi
    install -Dm755 "$phonon" "$libdir/libphonon.so"
    install -Dm644 pkg/zestengine.desktop $out/share/applications/zestengine.desktop
    install -Dm644 pkg/zestengine-hub.desktop $out/share/applications/zestengine-hub.desktop
    install -Dm644 pkg/zestengine.svg $out/share/icons/hicolor/scalable/apps/zestengine.svg
    for size in 16 32 48 64 128 256 512; do
      install -Dm644 "pkg/icons/zestengine-$size.png" \
        "$out/share/icons/hicolor/''${size}x''${size}/apps/zestengine.png"
    done
    install -Dm644 README.md $out/share/doc/zestengine/README.md
    install -Dm644 LICENSE $out/share/licenses/zestengine/LICENSE
    install -Dm644 NOTICE $out/share/licenses/zestengine/NOTICE
    install -Dm644 crates/zestengine-engine/assets/licenses/steam-audio-LICENSE.txt \
      $out/share/licenses/zestengine/steam-audio-LICENSE.txt
    install -Dm644 crates/zestengine-editor/assets/fonts/OFL.txt \
      $out/share/licenses/zestengine/OFL.txt
  '';

  passthru = {
    inherit runtimeLibs;
  };

  meta = {
    description = "3D game engine and editor";
    homepage = "https://github.com/lemonxah/zestengine";
    license = lib.licenses.unfree; # ZestEngine License 1.1 (source-available, see LICENSE)
    maintainers = [ ];
    platforms = lib.platforms.linux;
    mainProgram = "zestengine";
  };
}
