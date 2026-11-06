{
  description = "ZestEngine - 3D game engine and editor";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        zestengine = pkgs.callPackage ./default.nix { };
      in
      {
        packages.default = zestengine;

        devShells.default = pkgs.mkShell {
          inputsFrom = [ zestengine ];
          packages = with pkgs; [
            rust-analyzer
            clippy
            rustfmt
          ];
          env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath zestengine.runtimeLibs;
        };
      }
    );
}
