了解可用的工具及其工作原理非常重要。本节将介绍训练器和其他实用程序的各个组件。

# 组件列表
## 主要组件

* train.py - 训练网络的入口点
* model.py - 包含网络架构的定义并描述推理过程
* serialize.py - 处理在 model.py 中定义的模型的序列化/反序列化，以及 .nnue、.pt、.ckpt 序列化模型之间的转换。
* features.py, halfka.py, halfka_v2.py, halfka_v2_hm.py, halfkp.py - 网络输入的描述
* training_data_loader.cpp - 处理加载训练数据（.bin、.binpack）并准备批次
* nnue_dataset.py - 本机数据加载器和训练器之间的代理
* feature_transformer.py - 第一层（具有稀疏输入的完全连接层）的高度优化的 CUDA 实现

## 实用程序和工具

* cross_check_eval.py - 一个实用程序脚本，用于检查 pytorch 模型评估和使用 .nnue 模型的 stockfish 播放器评估之间的一致性
* visualize.py - 为单个网络[与基线相比]提供全面的可视化
* visualize_multi_hist.py - 为多个网络提供比较可视化
* run_games.py - 提供一种在训练进行中为生成的网络使用 c-chess-cli 运行游戏的方法
* do_plots.py - 创建训练期间收集的事件数据（训练损失、验证损失、ordo elo）的图表
* delete_bad_nets.py - 根据 ordo elo 删除最差的网络

# train.py

该组件是训练过程的入口点。它识别用于训练的设备，汇集训练参数，创建初始（或加载现有）模型，创建数据加载器（并定义时期大小），设置 tensorboard 记录器，并启动训练过程。

要查看可用的调用参数，可以运行 `python train.py --help`。

此训练器中时期的概念与通常的定义略有不同，在这里我们将一个时期定义为 1 亿个样本。这是因为训练数据集的大小可能差异很大。

除非另有说明（`--max_epochs`），否则训练过程是无限的。检查点以 .ckpt 格式保存在指定的日志目录中（请参阅调用参数）。何时以及保存哪些检查点可以通过更改传递给 `pl.callbacks.ModelCheckpoint` 的参数来修改代码。 .ckpt 格式的单个检查点存储模型和优化器状态，因此它们相当大，建议仅每隔几个时期保存一次。

# model.py

该模型被定义为名为 `NNUE` 的 `pl.LightningModule`。为了更好地理解模型的底层结构，建议阅读[此文档](https://github.com/glinscott/nnue-pytorch/blob/master/docs/nnue.md)。

该模型定义了推理过程并选择了要优化的参数。为了考虑量化，它在每个步骤后将某些层的权重裁剪到支持的范围。训练循环由 pytorch-lightning 处理，它通过模型的某些方法与模型交互。

该模型还包含允许在加载现有模型时在某些特征集之间进行转换的代码。值得注意的是，可以在加载现有模型后添加虚拟特征。该模型模块还公开了一个函数，用于将这些虚拟特征权重合并到真实特征权重中。

# serialize.py

虽然该组件被训练器用于加载现有模型，但它也可以用作模型格式转换的独立脚本。在播放引擎可以使用网络之前，必须将其转换为 .nnue 格式。它支持以下格式之间的转换：
* .nnue - 播放引擎使用的格式。它以量化形式包含网络，并且不保留优化器状态。
* .pt - pytorch 的格式。它不保留优化器状态，因此在尝试使用对优化器的更改重新启动训练时很有用。它使用全精度存储网络。
* .ckpt - 存储检查点的格式。训练器仅生成 .ckpt 格式的网络。像这样保存的网络包含完整的优化器状态和训练器状态，例如当前时期。此格式可用于临时暂停训练。

序列化器与模型紧密耦合。对模型的更改通常需要对序列化器进行相应的更改。此外，只有序列化器知道在转换为 .nnue 格式时如何执行量化，并且它必须与播放引擎中的量化实现相对应。

# features.py, halfka.py, halfka_v2.py, halfka_v2_hm.py, halfkp.py

`features.py` 文件导入所有可用的特征集。各个特征集定义了它们的组成部分（特征块）及其大小，为每个特征提供初始 PSQT 值，并允许检索每个特征的因子（与给定真实特征相对应的所有特征（真实或虚拟））。在 python 端，不必实现将棋盘状态分解为特征列表。

# training_data_loader.cpp

由于性能要求，有必要以本机方式实现训练数据加载器。可以使用 `compile_data_loader.bat`（使用 cmake）编译数据加载器。如果知道如何操作，也可以在没有 cmake 的情况下进行编译。

该组件实现了将棋盘状态分解为单个特征。此步骤必须与 python 端定义的特征集一致。

数据加载器一次提供整个批次，以消除稍后将单个样本连接成批次的成本。`SparseBatch` 表示这样一个批次。它为样本的每个属性分配一个数组。特征存储为活动特征的（索引，值）对。一个局面中活动特征数量的上限是已知的。未使用的特征存储时索引为 `-1`。

数据加载器支持多线程。可以在创建时指定线程数，并[启发式地]在从磁盘读取数据的线程和形成批次的线程之间分配。

数据加载器还支持形成 fen 字符串数组而不是训练批次。这对于其他实用程序（例如 `cross_check_eval.py`）很有用。

绑定是在 `nnue_dataset.py` 组件中使用 ctypes 完成的。

# nnue_dataset.py

包含与 pytorch 兼容的数据加载器的定义，并使用 ctypes 绑定使其使用本机数据加载器。批次中的张量被复制到主训练设备并重构成 pytorch 张量。

它要求已编译的本机数据加载器的共享库作为 `training_data_loader.[so|dll|dylib]` 存在于脚本目录中。

# feature_transformer.py

该文件包含网络第一层的 CUDA 实现（使用 CuPY）。我们以一种非常特殊的方式利用了非常稀疏的输入，与使用 pytorch 的稀疏张量相比，这大大加快了训练速度。有关此内容的更多信息可以在[此处](https://github.com/glinscott/nnue-pytorch/blob/master/docs/nnue.md#optimizing-the-trainer-cuda)找到。