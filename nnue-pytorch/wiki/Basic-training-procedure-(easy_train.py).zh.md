本节说明 `easy_train.py` 脚本的实际作用、特性以及如何有效使用。该脚本位于本仓库的 `script` 目录中，是继续操作所需的唯一文件。

# `easy_train.py` 是什么

`easy_train.py` 是一个 python 脚本，它以有组织且自包含的方式管理网络训练和网络测试。它旨在让用户只需最少的知识即可根据现有规范运行，同时也为有经验的用户保留了可定制性。输出被聚合和显示，既可以在 TUI（终端用户界面）中供人类用户使用，也可以直接输出到终端以允许管道操作。它已在 Windows (>=7) 和 Linux 上进行过测试。

![easy_train.py TUI](https://user-images.githubusercontent.com/8037982/176677909-7d36b6e1-6ac9-43f1-8965-c6394988cdca.png)

## 基本概念

该脚本将训练会话组织成“实验”。每个“实验”都会在本地设置训练器、播放器、日志记录和其他先决条件（例如启动网络模型），以便可以准确确定实验的运行方式。实验可以中途恢复，因此它们不完全是独立的会话。

尽管 `easy_train.py` 脚本位于 `nnue-pytorch` 仓库的 `scripts` 目录中，但它不需要仓库中的任何其他文件即可运行。它是一个单一的、自包含的 python 源文件。

## 依赖和要求

在 Windows 上，需要安装带有 GCC 的 MSYS2。可以按照 https://packages.msys2.org/groups/mingw-w64-x86_64-toolchain 轻松安装。编译器必须可以通过命令行直接调用 `gcc`，即其二进制文件所在的文件夹必须添加到 PATH 环境变量中。

待办事项：需要哪些全局可用的软件。阐明 msys2 的要求/设置

## 设置说明

### Windows

待办事项：从干净的系统安装开始，详细说明如何设置

### Linux

待办事项：从干净的系统安装开始，详细说明如何设置

## 工作区目录结构

该脚本在首次调用时在指定路径中创建一个工作区。该工作区具有以下目录树：

- **workspace_dir**
  - **books** - 包含所有至少使用过一次的 .epd 开局库。也可能以存档格式包含它们，具体取决于它们的获取方式。
  - **c-chess-cli** - 包含 [c-chess-cli](https://github.com/lucasart/c-chess-cli) 在特定提交的克隆，以最大程度地减少兼容性问题。它在首次使用时从源代码构建。
  - **ordo** - 包含 [ordo](https://github.com/michiguel/Ordo) 在特定提交的克隆，以最大程度地减少兼容性问题。它在首次使用时从源代码构建。
  - **experiments** - 包含所有训练会话（实验），每个会话一个目录
    - **experiment_1** - 包含单个实验的所有数据，此处实验名为“1”
      - **logging** - 包含 CLI 参数的转储和执行日志
      - **nnue-pytorch** - 包含指定训练器仓库的克隆。此脚本管理的所有任务都使用此目录中的脚本。数据加载器在首次使用时构建。
      - **start_models** - 如果训练基于其他实验或现有模型，则此目录将包含用于为训练提供种子的模型
      - **stockfish_base** - 包含包含基准游戏引擎的指定仓库的克隆。它应该是类似 stockfish 的。播放器在首次使用时从源代码构建。
      - **stockfish_test** - 包含包含与正在生成的网络相对应的游戏引擎的指定仓库的克隆。特别是，当网络架构更改时，它将是与基准不同的播放器。它应该是类似 stockfish 的。播放器在首次使用时从源代码构建。
      - **training** - 包含训练器输出，包括 tensorboard 事件、网络检查点、c-chess-cli 输出、ordo 输出
        - **run_0** - 包含来自单个 train.py 实例的输出，特别是 .nnue 格式的转换后网络
          - **lightning_logs**
            - **version_0** - 通常只会出现“0”版本。每次使用相同的日志目录调用训练时都会创建一个目录，这是 pytorch-lightning 的事情。此目录包含可由 tensorboard 读取的 tfevents 文件。
              - **checkpoints** - 包含训练的原始 .ckpt 完整检查点，包括优化器状态。这些可以转换为 .pt/.nnue 或直接用于进一步的训练。
        - **run_1** - 每个运行都有一个单独的目录，并且每个实验可以进行多次运行。如果是这种情况，则会出现多个“run_*”目录。
          - **...**
        - **...**
    - **experiment_2**
      - **...**
    - **...**

## 基本行为开关

待办事项：从头开始/继续/重新训练

## 示例调用

脚本目录中存在的示例调用（.bat 和 .sh）尚未准备好用于生产。它们的存在仅仅是为了显示最重要的可用选项并用于测试目的。以下是应该接近 master 的调用，并附有重要选项的注释 (.sh)。

第一次训练会话 - 从头开始训练网络：

```bash
python easy_train.py \
    --training-dataset=nodes5000pv2_UHO.binpack \ # 请参阅有关数据集的维基
    --validation-dataset=nodes5000pv2_UHO.binpack \ # 请参阅有关数据集的维基
    --num-workers=4 \ # 足以在几乎所有 gpu 上获得良好速度
    --threads=2 \ # 足以在几乎所有 gpu 上获得良好速度
    --gpus="0," \ # 仅使用第一个 gpu，目前没有多 gpu 训练；如果在此处指定了多个 gpu，则只会并行进行更多运行
    --runs-per-gpu=1 \ # 如果您的 gpu 未饱和，可以增加它
    --batch-size=16384 \
    --max_epoch=600 \ # 网络在 400 个 epoch 左右开始饱和
    --do-network-training=True \
    --do-network-testing=True \
    --tui=True \
    --network-save-period=10 \ # 每 10 个网络保存一次，您可以根据愿意使用的存储空间进行更改
    --random-fen-skipping=3 \
    --start-lambda=1.0 \
    --end-lambda=0.75 \
    --gamma=0.992 \ # 默认 gamma，确定每个 epoch 后学习率下降的速度
    --lr="8.75e-4" \ # 默认学习率
    --fail-on-experiment-exists=True \
    --build-engine-arch=x86-64-modern \ # 如果您的 CPU 支持，您可以将其更改为其他架构（这是 stockfish makefile 的 ARCH 参数）
    --build-threads=2 \
    --epoch-size=100000000 \ # 一个长期使用的非常标准的值，无需更改，而是非常 max_epoch
    --validation-size=1000000 \ # 验证步骤不是必需的，所以我们在这里做的工作很少
    --network-testing-threads=24 \ # 尽可能多地提供，会因内存压力而减慢训练速度
    --network-testing-explore-factor=1.5 \
    --network-testing-book="https://github.com/official-stockfish/books/raw/master/UHO_XXL_+0.90_+1.19.epd.zip" \
    --network-testing-nodes-per-move=20000 \ # 使用节点而不是时间以获得更一致的结果很重要
    --network-testing-hash-mb=8 \
    --network-testing-games-per-round=200 \
    --engine-base-branch=official-stockfish/Stockfish/master \ # 设置与您的训练运行相关的分支
    --engine-test-branch=official-stockfish/Stockfish/master \ # 设置与您的训练运行相关的分支
    --nnue-pytorch-branch=glinscott/nnue-pytorch/master \ # 设置与您的训练运行相关的分支
    --workspace-path=./easy_train_data \ # 更改为您希望存储所有数据的位置
    --experiment-name=test # 更改为您喜欢的任何名称，不要使用空格
```

第二次训练会话 - 使用更好的数据微调现有网络：

待办事项：需要验证

```bash
python easy_train.py \
    --training-dataset=Leela-dfrc_n5000.binpack \ # 请参阅有关数据集的维基
    --validation-dataset=Leela-dfrc_n5000.binpack \ # 请参阅有关数据集的维基
    --num-workers=4 \ # 足以在几乎所有 gpu 上获得良好速度
    --threads=2 \ # 足以在几乎所有 gpu 上获得良好速度
    --gpus="0," \ # 仅使用第一个 gpu，目前没有多 gpu 训练；如果在此处指定了多个 gpu，则只会并行进行更多运行
    --runs-per-gpu=1 \ # 如果您的 gpu 未饱和，可以增加它
    --start-from-experiment=test \ # 要从中重新训练的实验的名称，将选择最佳/最新的网络。或者使用 --start-from-model
    --batch-size=16384 \
    --max_epoch=800 \ # 网络在 400 个 epoch 左右开始饱和，但现在给它更多时间，因为我们使用的是较低的 LR 和较慢的计划
    --do-network-training=True \
    --do-network-testing=True \
    --tui=True \
    --network-save-period=10 \ # 每 10 个网络保存一次，您可以根据愿意使用的存储空间进行更改
    --random-fen-skipping=3 \
    --start-lambda=1.0 \
    --end-lambda=0.75 \
    --gamma=0.995 \ # 比第一次训练会话慢地降低 LR
    --lr="4.375e-4" \ # 开始 LR 的一半
    --fail-on-experiment-exists=True \
    --build-engine-arch=x86-64-modern \ # 如果您的 CPU 支持，您可以将其更改为其他架构（这是 stockfish makefile 的 ARCH 参数）
    --build-threads=2 \
    --epoch-size=100000000 \ # 一个长期使用的非常标准的值，无需更改，而是非常 max_epoch
    --validation-size=1000000 \ # 验证步骤不是必需的，所以我们在这里做的工作很少
    --network-testing-threads=24 \ # 尽可能多地提供，会因内存压力而减慢训练速度
    --network-testing-explore-factor=1.5 \
    --network-testing-book="https://github.com/official-stockfish/books/raw/master/UHO_XXL_+0.90_+1.19.epd.zip" \
    --network-testing-nodes-per-move=20000 \ # 使用节点而不是时间以获得更一致的结果很重要
    --network-testing-hash-mb=8 \
    --network-testing-games-per-round=200 \
    --engine-base-branch=official-stockfish/Stockfish/master \ # 设置与您的训练运行相关的分支
    --engine-test-branch=official-stockfish/Stockfish/master \ # 设置与您的训练运行相关的分支
    --nnue-pytorch-branch=glinscott/nnue-pytorch/master \ # 设置与您的训练运行相关的分支
    --workspace-path=./easy_train_data \ # 更改为您希望存储所有数据的位置
    --experiment-name=test_retrain # 更改为您喜欢的任何名称，不要使用空格
```
