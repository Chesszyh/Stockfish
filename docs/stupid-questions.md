# Stupid Environment Questions

1. 无法命中断点：`module containing this breakpoint has not yet loaded`

参考https://github.com/Microsoft/vscode-cpptools/issues/416：编译时添加`-g`选项。

```bash
make build debug=yes
```

2. clangd找不到系统头文件

利用`bear`创建`compile_commands.json`文件。

```bash
# You're in Stockfish root directory
bear -- make -C src build COMP=clang
```

然后重新加载VSCode窗口。

