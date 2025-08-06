# Stockfish 训练数据生成与处理工作流

本文档详细介绍了如何使用 Stockfish `tools` 分支中的工具来生成、处理和准备用于神经网络 (NNUE) 训练的数据集。整个流程涉及多个步骤，从初始数据生成到最终的文件准备。

## 编译工具

首先，你需要编译 `tools` 分支的 Stockfish。这通常可以通过在 `src` 目录下运行 `make` 命令来完成。请参考官方的编译指南以获取针对您操作系统的详细指令。

```bash
# 进入 src 目录
cd /home/chesszyh/Project/stockfish/src

# 编译 (以一个通用架构为例)
make -j build ARCH=x86-64-modern
```

编译成功后，你会在 `src` 目录下得到一个 `stockfish` 可执行文件。所有的 C++ 工具都通过这个文件以子命令的形式调用。

## 步骤 1: 生成训练数据

这是整个工作流的起点。我们使用 `generate_training_data` 命令通过引擎自对弈来创建数据。数据默认以高效的 `binpack` 格式存储。

**核心命令:**

```bash
./stockfish
```

进入 Stockfish 的交互式命令行后，执行：

```uci
generate_training_data count 1000000 output_file_name training_data
```

**说明:**

*   这个命令会生成包含 1,000,000 个局面的训练数据。
*   `output_file_name training_data` 指定了输出文件的前缀。由于默认格式是 `binpack`，最终文件将被命名为 `training_data.binpack`。
*   你可以根据需求调整 `count` 参数来控制数据量的大小。
*   强烈建议在交互式模式下运行此命令，以便在生成数据前设置 UCI 选项，例如线程数 (`setoption name Threads value 8`) 或关闭浅层裁剪 (`setoption name PruneAtShallowDepth value false`) 来提高数据质量。

**高级用法:**

*   **使用开局库**: 通过 `book` 参数指定一个 EPD 格式的开局库文件，可以让自对弈从不同的起始局面开始，增加数据多样性。
    ```uci
    generate_training_data book /path/to/your/book.epd count 1000000 ...
    ```
*   **控制随机性**: `random_move_count`、`random_move_min_ply` 和 `random_move_max_ply` 等参数可以控制对局中随机着法的数量和发生时机，以探索更多样的变化。

## 步骤 2: 数据转换 (可选)

虽然 `binpack` 是推荐的格式，但有时你可能需要将数据转换为其他格式，例如人类可读的 `.plain` 文本格式，或者用于兼容旧工具的 `.bin` 格式。

**核心命令:**

```uci
convert from_path training_data.binpack to_path training_data.plain
```

**说明:**

*   `from_path` 和 `to_path` 分别是输入和输出文件。
*   工具会根据文件扩展名 (`.binpack`, `.bin`, `.plain`) 自动推断格式。
*   如果你有来自其他来源的数据（例如 PGN 文件），可以使用 `script/pgn_to_plain.py` 脚本将其转换为 `.plain` 格式，然后再用 `convert` 命令转为二进制格式。

## 步骤 3: 重新评分 (Rescoring)

生成的数据通常是基于较浅的搜索深度。为了提高数据质量，可以使用 `transform rescore` 命令，用更深的搜索来重新评估数据集中的每一个局面。

**核心命令:**

```uci
transform rescore input_file training_data.binpack output_file training_data_rescored.binpack depth 10
```

**说明:**

*   `input_file` 指向原始数据文件。
*   `output_file` 是保存重评分数据的新文件。
*   `depth 10` 指定了用于重新评估的搜索深度。这是一个计算密集型操作，深度越高，耗时越长。
*   此工具会自动利用所有可用的 CPU 线程（遵循 `Threads` UCI 选项）。

## 步骤 4: 数据验证 (可选但推荐)

在进行下一步之前，最好验证一下数据的完整性，确保没有错误。

**核心命令:**

```uci
validate_training_data in_path training_data_rescored.binpack
```

**说明:**

*   如果文件有效，该命令会安静地执行完毕。如果检测到任何问题（如非法局面或着法），它会报告错误。

## 步骤 5: 打乱数据 (Shuffle)

这是训练前**至关重要**的一步。原始数据是按对局顺序排列的，具有很强的相关性。必须将其完全打乱，才能用于神经网络的有效训练。

**核心工具:** `script/shuffle_binpack.py`

这是一个 Python 脚本，需要独立运行。

```bash
# 退出 Stockfish 交互环境，回到 shell
python /home/chesszyh/Project/stockfish/tools/script/shuffle_binpack.py training_data_rescored.binpack training_data_shuffled.binpack
```

**说明:**

*   第一个参数是输入文件，第二个参数是输出文件。
*   这个脚本会消耗大量内存，因为它需要将整个数据集加载到 RAM 中进行打乱。如果你的数据集非常大，请确保有足够的内存。

## 步骤 6: 准备训练

现在，你已经拥有了一个经过生成、重评分和打乱的高质量数据集 `training_data_shuffled.binpack`。这个文件可以直接用于 Stockfish 的 NNUE 训练流程。

你通常还需要一个独立的、较小的数据集作为**验证集 (validation set)**，用于在训练过程中评估模型的性能。验证集的生成过程与训练集完全相同，但应使用不同的随机种子 (`seed` 参数) 和/或不同的开局库，以确保其与训练集没有重叠。

## 总结工作流

1.  **编译**: `make`
2.  **生成**: `generate_training_data` -> `data.binpack`
3.  **重评分**: `transform rescore` -> `data_rescored.binpack`
4.  **打乱**: `python shuffle_binpack.py` -> `data_shuffled.binpack`
5.  **训练**: 使用 `data_shuffled.binpack` 作为训练集，并准备一个独立的验证集。

## 最佳调参实践：压榨硬件性能获取高质量数据集

针对您的机器配置（CPU: i9-13900HX 32逻辑核心，16GB RAM；GPU: NVIDIA GeForce RTX 4060），以下是优化 Stockfish 训练数据生成过程的建议：

Command: 

```bash
setoption name Threads value 32
setoption name Hash value 8192
setoption name PruneAtShallowDepth value false

generate_training_data depth 5 count 1000000 # default count is 800000000, too large too long for 16GB RAM
transform rescore input_file training_data.binpack output_file training_data_rescored.binpack depth 10 # 重新深度评估局面，提高准确性
```

### 1. CPU 优化 (i9-13900HX)

Stockfish 的数据生成和重评分过程是高度 CPU 密集型的，可以充分利用多核处理器。

*   **线程数 (`Threads`)**: 这是最重要的参数。您的 i9-13900HX 拥有 32 个逻辑核心，理论上可以支持更多线程。在 `stockfish` 交互式命令行中，将 `Threads` 设置为 CPU 的逻辑核心数，甚至略高一些（例如 32 或 48），以观察性能。过高的线程数可能导致上下文切换开销，反而降低效率，需要根据实际测试找到最佳值。
    ```uci
    setoption name Threads value 32 # 或 48，根据实际测试调整
    ```
*   **哈希表大小 (`Hash`)**: 哈希表用于存储引擎的搜索结果，可以显著提高搜索效率。您的 16GB RAM 允许设置一个较大的哈希表。建议将其设置为 RAM 的一半左右，例如 8192 MB (8GB)。
    ```uci
    setoption name Hash value 8192 # 单位为 MB
    ```
*   **`PruneAtShallowDepth`**: 建议将其设置为 `false`，以确保在浅层搜索时不会进行剪枝，从而提高生成数据的质量和多样性。
    ```uci
    setoption name PruneAtShallowDepth value false
    ```

### 2. 内存优化 (16GB RAM)

除了哈希表，内存主要影响 `shuffle_binpack.py` 脚本的性能。

*   **`shuffle_binpack.py`**: 这个 Python 脚本在打乱数据时会将整个文件加载到内存中。对于 16GB RAM，您可以处理的数据集大小是有限的。如果生成的数据集文件过大（例如超过 10GB），`shuffle_binpack.py` 可能会因为内存不足而失败或变得非常慢（开始使用磁盘交换）。
    *   **解决方案**: 如果数据集过大，可以考虑分批生成和打乱，或者使用 `split_count` 参数将打乱后的数据分割成多个小文件，然后通过 `interleave_binpacks.py` 合并。

### 3. GPU 作用 (RTX 4060)

*   **数据生成阶段**: Stockfish 的数据生成（`generate_training_data` 和 `transform rescore`）是纯 CPU 密集型任务，**不会使用您的 RTX 4060 GPU**。GPU 在此阶段处于空闲状态。
*   **NNUE 训练阶段**: 您的 RTX 4060 GPU 将在后续的 NNUE 模型训练阶段发挥关键作用，显著加速模型的训练速度。因此，虽然它不参与数据生成，但对于整个 NNUE 工作流来说是必不可少的。

### 4. 数据质量与生成速度的平衡

*   **`generate_training_data` 的 `depth`**: 默认深度为 3。为了获得更高质量的数据，可以适当增加这个深度，例如 `depth 5` 或 `depth 6`。更高的深度意味着更准确的评估，但也会显著增加生成时间。
    ```uci
    generate_training_data depth 5 ...
    ```
*   **`transform rescore` 的 `depth`**: 这是提高数据质量的关键步骤。建议将此处的深度设置得更高，例如 `depth 10` 或 `depth 12`。这是对已生成局面进行更深层次的评估，以获得更精确的标签。
    ```uci
    transform rescore ... depth 12
    ```
*   **随机性参数**: 适当调整 `random_move_count`、`random_move_min_ply`、`random_move_max_ply` 和 `random_multi_pv_diff` 可以增加生成数据的多样性，这对于训练出更鲁棒的模型很有帮助。
    *   例如，`random_multi_pv_diff 100` 可以让引擎在选择随机着法时，考虑那些与最佳着法差距在 100 步兵值以内的次优着法，从而增加数据中“有趣”局面的比例。

### 5. 综合建议

1.  **始终在交互式模式下运行 Stockfish**：这样可以预先设置 UCI 选项（`Threads`, `Hash`, `PruneAtShallowDepth`），确保它们在数据生成和重评分过程中生效。
2.  **从小规模测试开始**：在投入大量时间生成数据之前，先用较小的 `count` 值和较浅的 `depth` 进行测试，验证整个流程是否顺畅，并观察资源使用情况。
3.  **监控资源**：在数据生成和打乱过程中，使用系统监控工具（如 `htop`、`nvidia-smi`）观察 CPU 利用率、内存使用情况和磁盘 I/O，以便及时发现瓶颈并进行调整。
4.  **分阶段生成**：如果目标数据集非常大，可以考虑分批生成数据（使用 `save_every` 参数），然后使用 `interleave_binpacks.py` 合并，以避免单次操作的内存压力过大。

通过以上调优，您应该能够最大限度地利用您的硬件，高效地生成高质量的 Stockfish NNUE 训练数据集。