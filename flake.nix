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
        config.allowBroken = true;
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [

          cudatoolkit_11 # CUDA 11.8

          ffmpeg-full
          python310
          python310Packages.pip
          # install in virtual env to get the specific version
          # compatible with the pytorch with adequate compute capability
          # python310Packages.numpy
          python310Packages.av

          stdenv.cc.cc.lib # required by numpy
          zlib
          glib # required by opencv-python: glib and glib.out

          # install in virtualenv
          # -> nix will install pytorch for a specific CUDA version which 
          # does not necessarily have the compute capability required by the device
          # python310Packages.mmengine
          # python310Packages.mmcv
          # python310Packages.torch-bin
          # python310Packages.torchvision-bin
        ];

        shellHook = ''
          echo "You are now using a NIX environment"
          export CUDA_PATH=${pkgs.cudatoolkit_11}
          export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.libGL}/lib:${pkgs.glib}/lib:${pkgs.glib.out}/lib:$LD_LIBRARY_PATH"
  	  # Add host NVIDIA libraries
  	  export LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/opengl-driver-32/lib:$LD_LIBRARY_PATH"
  	  export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
        '';
      };
    };
}
