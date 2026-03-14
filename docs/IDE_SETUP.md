# IDE Setup

This repository now includes a CMake-based development setup for both CLion and VSCode.

## CLion

1. Open `/home/chesszyh/Project/Chess/Stockfish` as a CMake project.
2. Let CLion load `CMakePresets.json`.
3. Pick the `dev-debug` preset for normal debugging or `dev-release` for optimized builds.
4. Use the `stockfish` target directly. Build and debug work out of the box.

## VSCode

Recommended extensions:

- `ms-vscode.cmake-tools`
- `llvm-vs-code-extensions.vscode-clangd`
- `ms-vscode.cpptools`

The repository includes:

- `.vscode/tasks.json`: `Ctrl+Shift+B` builds the `dev-debug` preset.
- `.vscode/launch.json`: `F5` builds and starts `build/dev-debug/stockfish`.
- `.vscode/settings.json`: CMake presets are enabled and code navigation is driven by `compile_commands.json`.
- `.vscode/c_cpp_properties.json`: cpptools fallback configuration.

For the closest CLion-like navigation experience in VSCode, use `clangd` as the language server and keep the active configure preset on `dev-debug`.
