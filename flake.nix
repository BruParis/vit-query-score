# This flake provides a skeleton dev environment for PyTorch with CUDA support /
#
# To test python:
# $ nix develop
# $ python
# >>> import torch
# >>> torch.cuda.is_available()
# >>> torch.cuda.device_count()
# >>> torch.cuda.get_device_name(0)

{
  description = "A flake providing a dev shell for PyTorch with CUDA and CUDA development using NVCC.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux"; # Adjust if needed
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [

          ffmpeg-full
          python312
          python312Packages.pip
          python312Packages.numpy
          python312Packages.mmengine
          python312Packages.mmcv
          python312Packages.torch-bin
          python312Packages.torchvision-bin
          python312Packages.av
        ];


        shellHook = ''
          echo "You are now using a NIX environment"
          export CUDA_PATH=${pkgs.cudatoolkit}
        '';
      };
    };
}
