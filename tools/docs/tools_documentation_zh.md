# Stockfish 工具文档

本文档概述了 Stockfish 工具库中可用的各种工具。这些工具主要用于为 Stockfish 国际象棋引擎生成、操作和验证训练数据。

## C++ 工具 (位于 `src/tools`)

这些工具通过 `./stockfish` 可执行文件后跟一个命令来调用。

### `convert`

在不同格式之间转换训练数据。

*   **`convert_bin_from_pgn_extract`**: 将 PGN 提取的数据转换为二进制格式。
*   **`convert_bin`**: 将文本格式转换为二进制格式。
*   **`convert_plain`**: 将二进制格式转换为纯文本格式。

### `generate_training_data`

从一组给定的开局局面生成训练数据。

### `generate_training_data_nonpv`

在不使用主变搜索的情况下生成训练数据。

### `opening_book`

管理 EPD 格式的开局库。

### `sfen_packer`

打包和解包 SFEN (日本将棋局面表示法) 局面。

### `stats`

从训练数据中收集统计信息。

### `transform`

对训练数据执行各种转换。

### `validate_training_data`

验证训练数据的完整性。

## Python 脚本 (位于 `script`)

这些脚本使用 Python 解释器执行。

### `extract_bin.py`

从二进制训练数据文件中提取指定数量的局面。

**用法:**

```bash
python extract_bin.py <文件名> <偏移量> <数量> [次数]
```

*   `<文件名>`: `.bin` 文件的路径。
*   `<偏移量>`: 从文件开头跳过的局面数量。
*   `<数量>`: 要提取的局面数量。
*   `[次数]` (可选): 在输出文件中重复提取的局面的次数。

### `interleave_binpacks.py`

将多个二进制训练数据文件合并为一个文件，并交错来自输入文件的数据。

**用法:**

```bash
python interleave_binpacks.py <输入文件1> ... <输入文件N> <输出文件>
```

*   `<输入文件1>` ... `<输入文件N>`: 输入的 `.bin` 文件。
*   `<输出文件>`: 输出文件的名称。

### `pgn_to_plain.py`

将 PGN 文件转换为可与 `learn convert_bin` 命令一起使用的纯文本格式。

**用法:**

```bash
python pgn_to_plain.py --pgn "<pgn文件>" --start_ply <回合数> --output <输出文件>
```

*   `--pgn`: 输入 PGN 文件的 glob 模式 (例如, `"data/*.pgn"`)。
*   `--start_ply`: 开始提取局面的回合数。
*   `--output`: 输出文本文件的名称。

### `shuffle_binpack.py`

打乱二进制训练数据文件中的局面。

**用法:**

```bash
python shuffle_binpack.py <输入文件> <输出文件> [分割数量]
```

*   `<输入文件>`: 输入的 `.bin` 文件。
*   `<输出文件>`: 输出的 `.bin` 文件。
*   `[分割数量]` (可选): 将打乱后的数据分割成的文件数量。
