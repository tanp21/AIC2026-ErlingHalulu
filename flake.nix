{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [ pkgs.uv pkgs.python312 pkgs.aria2 pkgs.ffmpeg pkgs.zlib pkgs.git-lfs ];

        shellHook = ''
          export LD_LIBRARY_PATH="${pkgs.zlib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          if [ ! -d .venv ]; then
            uv venv --python python3.12
          fi
          source .venv/bin/activate
          uv sync
        '';

        UV_LINK_MODE = "copy";
      };
    };
}
