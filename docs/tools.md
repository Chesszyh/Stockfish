# Stockfish tools 

1. Stockfish Tools 项目与代码结构

分支的结构清晰，主要围绕着为神经网络（NNUE）生成和处理训练数据这一核心功能。其关键目录和代码结构如下：

* `src/tools/`: 这是所有核心 C++ 工具的源代码所在地。这些工具被编译进 stockfish
    主程序，并以命令行子命令的形式调用（例如 stockfish generate_training_data ...）。主要工具包括：
    * generate_training_data.cpp: 用于通过引擎自对弈生成训练数据。这是最核心的功能之一。
    * convert.cpp: 在不同数据格式（如 .binpack, .bin, .plain）之间进行转换。
    * transform.cpp: 对训练数据进行变换，例如使用更深的搜索重新评估局面（rescore）。
    * validate_training_data.cpp: 验证训练数据文件的完整性和正确性。
    * stats.cpp: 从数据集中收集统计信息。

* `script/`: 包含一系列 Python 脚本，用于辅助 C++ 工具完成一些额外的数据处理任务。这些脚本需要独立于 stockfish
    主程序，使用 Python 解释器运行。主要脚本包括：
    * shuffle_binpack.py: 随机打乱数据集中局面的顺序，这对于训练至关重要。
    * interleave_binpacks.py: 将多个数据文件交错合并成一个。
    * pgn_to_plain.py: 将 PGN 格式的棋谱文件转换为 tools 可处理的 .plain 纯文本格式。
    * extract_bin.py: 从二进制数据文件中提取一部分局面。

* `docs/`: 存放所有工具的 Markdown 格式说明文档。每个文档详细介绍了对应工具的功能和使用参数。

* `data/`: 通常用于存放训练数据文件，例如 filtered.txt 可能是某种过滤后的局面集合。

总的来说，整个项目的工作流程是：使用 src/tools/ 中的 C++ 工具生成、转换和处理大规模数据，然后利用 script/ 中的 Python
脚本进行一些灵活的、文件级别的操作（如打乱、合并），而 docs/ 则为这一切提供了文档支持。