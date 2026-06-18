{
  description = "qwen3-tts.cpp — dev environment (C++/GGML, CPU + Vulkan)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Vulkan toolchain needed to build ggml with -DGGML_VULKAN=ON:
        # glslc/glslangValidator compile the .comp shaders, the loader +
        # headers provide the API, spirv-tools links the shader pipeline.
        vulkanInputs = [
          pkgs.vulkan-headers
          pkgs.vulkan-loader
          pkgs.shaderc          # glslc
          pkgs.glslang
          pkgs.spirv-tools
          pkgs.spirv-headers
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
            pkgs.git
            # Runtime/diagnostic helpers
            pkgs.vulkan-tools     # vulkaninfo
          ] ++ vulkanInputs;

          # ggml's CMake finds Vulkan via the standard module; make sure the
          # loader is also on the runtime path so the built CLI/libs resolve
          # libvulkan when GGML_VULKAN is enabled.
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.vulkan-loader ];

          shellHook = ''
            echo "qwen3-tts.cpp dev shell"
            echo "  cmake:  $(cmake --version | head -1)"
            echo "  glslc:  $(glslc --version 2>/dev/null | head -1)"
            echo
            echo "Build ggml (CPU+Vulkan):"
            echo "  cmake -S ggml -B ggml/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON"
            echo "  cmake --build ggml/build -j"
            echo "Build qwen3-tts:"
            echo "  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release"
            echo "  cmake --build build -j"
          '';
        };
      });
}
