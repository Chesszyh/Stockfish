# validate_training_data (验证训练数据)

`validate_training_data` 命令允许验证 `.plain`、`.bin` 和 `.binpack` 类型的训练数据。

与 Stockfish 中的所有命令一样，`validate_training_data` 既可以从命令行调用（例如 `stockfish.exe validate_training_data ...`），也可以在交互式提示符中调用。

此命令的语法如下：
```
validate_training_data in_path
```

`in_path` 是要验证的文件的路径。数据类型根据其扩展名（`.plain`、`.bin` 或 `.binpack` 之一）自动推断。
