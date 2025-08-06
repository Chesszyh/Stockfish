# NNUE 入门指南

## 1. NNUE 简介

NNUE（ƎUИИ，Efficiently Updatable Neural Network）是一种高效可更新的神经网络架构。它最初由日本的计算机将棋（Shogi）开发者那须悠（Yu Nasu）发明，并于 2018 年集成到 YaneuraOu 引擎中。后来，野田久顺（Hisayori Noda）于 2019 年将其移植到国际象棋领域的 Stockfish 引擎中。

NNUE 的引入是国际象棋引擎发展史上的一个里程碑。它成功地将神经网络的评估能力与传统 Alpha-Beta 搜索引擎的速度相结合，极大地提升了引擎的棋力。

## 2. NNUE 原理

NNUE 的核心思想在于利用棋盘游戏后续局面之间输入的微小变化。其主要原理包括：

*   **高效可更新**：国际象棋中，一步棋通常只会改变棋盘上少数棋子的位置。NNUE 利用这一点，通过增量更新（Incremental Updates）的方式，只计算网络中发生变化的部分，而不是每次都重新计算整个网络，从而大大提高了评估速度。
*   **稀疏输入**：NNUE 的输入特征是稀疏的。例如，一个特征可以表示为“白王在 e1，黑车在 d4”。在一个给定的局面中，只有少数特征是激活的（非零）。这使得第一层的计算可以被极大地优化。
*   **量化（Quantization）**：为了在 CPU 上实现极高的评估速度，NNUE 使用低精度整数（如 int8 和 int16）进行计算，而不是传统的浮点数。这个过程被称为量化。虽然量化会引入微小的精度损失，但对于 NNUE 这种相对较浅的网络来说，这种损失可以忽略不计，而带来的速度提升是巨大的。

## 3. NNUE 架构

一个典型的 NNUE 网络通常比较浅，通常只有 3-4 层。其结构如下：

1.  **特征转换器（Feature Transformer）**：这是网络的第一层，也是最大的一层。它接收稀疏的输入特征，并将其转换为一个较小的、密集的向量。这一层的计算得益于输入的稀疏性，并且可以通过“累加器（Accumulator）”机制进行高效的增量更新。
2.  **隐藏层（Hidden Layers）**：通常是 1 到 2 个全连接（Fully Connected）的线性层，使用 ClippedReLU 作为激活函数。ClippedReLU（`min(max(x, 0), 1)`）因其计算简单且适合量化而被广泛使用。
3.  **输出层（Output Layer）**：一个线性的全连接层，输出一个单一的数值，代表当前局面的评估值（通常以厘兵为单位）。

### 累加器（Accumulator）

累加器是实现增量更新的关键。它存储了第一层隐藏神经元的激活值。当棋盘上一个棋子移动时，只有少数输入特征会发生变化（一些特征被移除，一些特征被添加）。我们只需要从累加器中减去被移除特征对应的权重，并加上被添加特征对应的权重，就可以得到新的累加器值，而无需重新计算整个第一层。

## 4. NNUE 训练

训练 NNUE 网络的目标是最小化一个损失函数，该函数衡量网络输出的评估值与“真实”评估值之间的差距。

### 训练数据

高质量的训练数据至关重要。这些数据通常来自于：

*   **Stockfish 生成**：使用 Stockfish 的数据生成工具，通过自我对弈产生大量的局面和评估值。
*   **Lc0 数据转换**：将 Leela Chess Zero (Lc0) 的训练数据转换为 NNUE 可用的格式。通常认为 Lc0 的数据质量更高。

### 训练过程

本仓库提供了两种主要的训练脚本：

*   `train.py`：功能更全面的训练脚本，提供了丰富的可调参数。
*   `easy_train.py`：一个封装了训练和测试流程的脚本，更易于上手。

一个典型的训练命令如下所示：

```bash
python3 train.py \
    <path_to_training_data> \
    <path_to_validation_data> \
    --gpus "0," \
    --features=HalfKAv2_hm^ \
    --max_epochs=400 \
    --default_root_dir ./training/runs/run_0
```

### 损失函数

常用的损失函数包括：

*   **均方误差（Mean Squared Error, MSE）**：计算网络输出与目标值之差的平方。
*   **交叉熵（Cross Entropy）**：通常用于分类问题，但在 NNUE 训练中也表现良好。

训练时，通常会将局面的评估值（CP-space）通过 Sigmoid 函数转换到 WDL-space（0=负，0.5=和，1=胜），并结合比赛的实际结果（Win/Loss）来计算损失。

## 5. NNUE 使用

训练完成后，网络以 `.ckpt` 格式保存，其中包含了模型的权重、优化器状态等信息。要将网络用于国际象棋引擎（如 Stockfish），需要将其转换为 `.nnue` 格式。

本仓库提供了 `serialize.py` 脚本来完成格式转换：

```bash
# 将 .ckpt 转换为 .nnue
python serialize.py --features=<feature_set> model.ckpt model.nnue

# 将 .nnue 转换为 .pt (PyTorch 格式)
python serialize.py --features=<feature_set> model.nnue model.pt
```

`.nnue` 文件是一个量化后的、只包含网络权重的文件，体积小，加载快，可以直接被引擎使用。

## 6. NNUE 代码结构

本仓库的关键文件及其功能如下：

*   `model.py`：定义了 NNUE 网络的架构。
*   `train.py`：主要的训练脚本。
*   `easy_train.py`：简化的训练与测试管理脚本。
*   `serialize.py`：用于在 `.ckpt`, `.pt`, `.nnue` 格式之间转换模型。
*   `features.py`, `halfkp.py`, ...：定义了不同的输入特征集。
*   `training_data_loader.cpp`：使用 C++ 实现的高性能数据加载器。
*   `nnue_dataset.py`：数据加载器的 Python 封装，与 PyTorch 对接。
*   `feature_transformer.py`：第一层特征转换器的高度优化的 CUDA 实现。

## 7. NNUE 测试

评估一个训练好的 NNUE 网络是否优秀，最终需要通过实战来检验。

*   `run_games.py`：一个实用工具，可以使用 `c-chess-cli` 来运行引擎间的比赛，以测试训练过程中生成的网络的棋力。
*   `cross_check_eval.py`：用于交叉验证 PyTorch 模型和 `.nnue` 模型对同一局面的评估值是否一致。
*   `visualize.py`：提供网络的可视化分析。

通常，我们会让新训练出的网络与当前最好的网络进行数千盘甚至上万盘的对局，通过 Elo 等级分的变化来判断新网络的棋力。
