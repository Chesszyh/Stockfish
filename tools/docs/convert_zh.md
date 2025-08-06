# Convert (转换)

`convert` 命令允许在 `.plain`、`.bin` 和 `.binpack` 格式之间转换训练数据。

与 Stockfish 中的所有命令一样，`convert` 既可以从命令行调用（例如 `stockfish.exe convert ...`），也可以在交互式提示符中调用。

此命令的语法如下：
```
convert from_path to_path [append] [validate]
```

`from_path` 是要转换的源文件路径。数据类型根据其扩展名（`.plain`、`.bin` 或 `.binpack` 之一）自动推断。
`to_path` 是输出文件的路径。数据类型根据其扩展名推断。如果文件不存在，将会被创建。

`append` 和 `validate` 是可选参数，可以按任意顺序出现。
如果未指定 `append`，输出文件在写入任何内容之前将被清空。如果指定了 `append`，转换后的训练数据将被追加到输出文件的末尾。

如果指定了 `validate`，转换过程将在发现第一个非法着法时停止，并显示一条诊断信息。
