# NNUE训练

## 部分依赖

- CuPy：NumPy 风格的 GPU 数组库（CUDA 后端），在数据预处理/特征计算时可用到 GPU 加速。
- PyTorch Lightning：高层训练框架，封装训练循环/日志/多设备分发，简化分布式与回调。
- GPUtil：Python 库，查询 GPU 使用率/显存，便于 easy_train 自动选择空闲 GPU 或做资源提示。

## 环境准备

以Ubuntu为例：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git unzip pkg-config libomp-dev wget curl zip

# 创建并启用环境（建议 3.10/3.11）
conda create -y -n nnue python=3.10
conda activate nnue

# 安装 CuPy（匹配 CUDA 版本的 wheel，二选一）
pip install cupy-cuda12x        # Cupy 统一 12x wheels（新版本）
# 或： pip install cupy-cuda121  # 严格 12.1
# 或： pip install cupy-cuda124  # 严格 12.4

# 训练与工具依赖: requirements.txt在项目根目录下
pip install -r requirements.txt

# 可选：验证 CUDA 是否可用
python - <<'PY'
import torch, cupy, GPUtil, sys
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "is_available:", torch.cuda.is_available())
print("CuPy :", cupy.__version__, "CUDA runtime:", cupy.cuda.runtime.runtimeGetVersion())
gpus = GPUtil.getGPUs()
print("GPUs :", [(g.id, g.name, f"{g.memoryUsed}/{g.memoryTotal}MB") for g in gpus])
PY
```

运行时，可通过`watch -n 1 nvidia-smi`或[nivtop](https://github.com/XuehaiPan/nvitop)来监控 GPU 使用情况。

# 启动训练

- easy_train.py 做了什么（总体流程）
- example.sh 每个参数对应 easy_train.py/训练脚本里的含义
- RTX 4060 8GB 的实用调参建议（含一条推荐命令）

## easy_train.py 流程概览
- 依赖检查与环境准备
  - 外部：cmake/make/gcc，GPU/CUDA，PyTorch+CUDA，cupy，pytorch_lightning，GPUtil。
  - 下载并编译：c-chess-cli；可选 ordo；编译 nnue-pytorch 的数据加载器（compile_data_loader.bat 用 sh 调起）。
  - 下载开局库（.epd.zip），解压到工作区。
  - 可选编译两份 Stockfish（base/test）用于对弈测试。
- 训练
  - 按 --gpus 和 --runs-per-gpu 生成多个 TrainingRun 线程；每个线程起一个独立的 python train.py 子进程。
  - 给 train.py 传入：数据集、batch、线程数、跳样策略、学习率/衰减、lambda/其调度、特征集、epoch-size、validation-size、保存周期、起始模型/断点等。
  - 解析 Lightning 进度条，记录 epoch/step/loss/it/s，检测 OOM，写日志与 TUI。
- 测试
  - 独立线程起 python run_games.py；并行对弈、收集结果（可选 ordo），把最优网络汇总显示。
- 退出与守护
  - 可设置超时退出/训练结束后延时退出；确保子进程退出；保存日志。

## example.sh 参数逐项解读（它们如何被 easy_train.py 使用）
训练相关（传给 train.py）
- --training-dataset/--validation-dataset：训练/验证 .binpack 路径。easy_train.py 会校验文件存在，并把这些路径直接拼到 train.py 命令行。
- --num-workers=4：DataLoader 进程数（供训练数据），影响数据吞吐。
- --threads=2：PyTorch 线程数，CPU 端 BLAS/OpenMP 并行度。
- --gpus="0,"：仅用于选择 GPU ID 列表。脚本内部会解析出 [0]，并为每个 run 传 --gpus=<单个ID>, 给 train.py。
- --runs-per-gpu=2：同一张 GPU 上并行跑 2 个训练子进程（每个各自有独立 batch 和数据流）。
- --batch-size=16384：每个训练子进程的 batch。多个并行 run 会叠加显存占用。
- --max_epoch=10：训练 epoch 数。
- --network-save-period=1：每个 epoch 存一个 checkpoint，IO 频繁但利于回溯。
- --random-fen-skipping=3：随机跳样（平均跳过 3 个取 1 个），提升样本去相关/多样性。
- --start-lambda=1.0/--end-lambda=0.75/--lambda（默认）：评价与结果的插值系数调度（lambda 从 1.0 线性走到 0.75），对应文档里的 WDL 插值。
- --epoch-size=1638400/--validation-size=16384：每个 epoch/验证步使用的局面数，直接决定每个 epoch 的时长。
- --lr=8.75e-4/--gamma=0.992：初始学习率、每 epoch 的乘性衰减。
- --features="HalfKAv2_hm^"：特征集名称，传给 train.py（以及 serialize.py 等转换脚本）。
- --additional-training-arg="...": 直接原样追加到 train.py（可用于传 Lightning/Trainer 的原生参数，如支持的话）。

测试与基础设施
- `--do-network-training=True`：是否训练（关掉可只跑测试/搭环境）。
- --do-network-testing=True：是否边训边测。开启会占用较多 CPU（对弈是 CPU 密集）。
- --network-testing-threads=24：run_games.py 并行对弈并发度（强占 CPU）。越大越快，但会与训练抢 CPU。
- --network-testing-explore-factor=1.5：选择测试候选时放大 Elo 误差的系数（探索 vs 利用）。
- --network-testing-book=...epd.zip：开局库下载地址。
- --network-testing-nodes-per-move=20000：每步节点数（推荐优于时控，稳定）。
- --network-testing-hash-mb=8：每个引擎分配的置换表大小 MiB。
- --network-testing-games-per-round=200：每轮自对弈局数。
- --engine-base-branch/--engine-test-branch：Stockfish 基线/被测分支（repo/branch_or_commit），会拉源码、编译。
- --build-engine-arch=x86-64-bmi2/--build-threads=2：引擎编译架构与并行度。
- --nnue-pytorch-branch=vondele/nnue-pytorch/easy_train：训练器分支（会下载并编译数据加载器）。
- --workspace-path=.../--experiment-name=test：工作区与实验名（目录布局：experiments/experiment_<name>/...）。
- --fail-on-experiment-exists=False：实验目录存在时不报错（搭配 --resume-training）。
- --tui=True：终端 UI。远程/日志场景可关掉。
- --additional-testing-arg="...": 原样传给 run_games.py。

## 在 RTX 4060 8GB 上的实用调参建议
目标：在不 OOM 的前提下最大化 GPU 利用率，同时避免 CPU/IO 成瓶颈。

显存与并发
- runs-per-gpu：建议 1 开始。8GB 上并发 2 个 run 很容易 OOM；若确需 2 个，把每个 run 的 batch-size 砍半。
- batch-size：
  - 起步 16384；观察 nvidia-smi 显存与 easy_train 的 OOM 日志。
  - 若显存富余，尝试 24576 或 32768；一旦 OOM，回退到上一个稳定值再下调 10~20%。
  - 若想“等效”放大 batch 又不想增显存，可用梯度累计（前提：train.py/Lightning 支持），示例见下。
数据供给与 CPU
- num-workers：提升到 8~16（取决于 CPU 核心/磁盘）。若 GPU 利用率低而 CPU 空闲，增大此值；若系统负载高，降低。
- threads（PyTorch 线程）：2~4 一般足够，过高反而抢 DataLoader/测试的 CPU。
- 数据放在 NVMe SSD，避免网络/机械盘瓶颈。

测试并发与资源隔离
- do-network-testing：初训阶段建议关闭，或把 --network-testing-threads 压到 4~8，避免和 DataLoader 抢 CPU。
- nodes-per-move：20000 较重，若 CPU 忙可降到 10000 或减少 games-per-round。

保存与 IO
- network-save-period：调大到 5~10，降低每个 epoch 的 IO 开销（例子里设 1 会频繁写盘）。
- resume-training=True：保持默认可断点续训。

学习率与插值
- lr/gamma：默认已保守稳定（8.75e-4/0.992）。若 batch 明显变大，可略提 lr；否则维持默认。
- start-lambda/end-lambda：建议保留从 1.0 → 0.75 的线性过渡；数据/任务不同可试 1.0 → 0.5。

可选（仅在 train.py/Lightning 支持时）
- 混合精度：添加
  - --additional-training-arg="--precision=bf16"（Ada 支持，CUDA 12/驱动较新更稳）
  - 或 --additional-training-arg="--precision=16"（fp16）
  - 小网络上收益有限但可试。
- 梯度累计：如支持 Lightning/Trainer
  - --additional-training-arg="--accumulate_grad_batches=2"
  - 等效把 batch*2，几乎不增显存但会减小 it/s。

监控与迭代
- 用 nvidia-smi、日志中的 it/s、kpos/s（脚本已显示）综合判断瓶颈是 GPU、CPU 还是 IO。
- 出现 "CUDA out of memory" 时脚本会终止该 run；据此下调 batch 或减少并发。

一条面向 RTX 4060 8GB 的推荐命令（稳妥起步）
````bash
python easy_train.py \
  --training-dataset=/path/to/train.binpack \
  --validation-dataset=/path/to/val.binpack \
  --gpus="0," \
  --runs-per-gpu=1 \
  --batch-size=16384 \
  --num-workers=8 \
  --threads=2 \
  --max_epoch=10 \
  --epoch-size=2000000 \
  --validation-size=20000 \
  --lr=8.75e-4 \
  --gamma=0.992 \
  --start-lambda=1.0 \
  --end-lambda=0.75 \
  --features="HalfKAv2_hm^" \
  --network-save-period=5 \
  --do-network-testing=False \
  --fail-on-experiment-exists=False \
  --workspace-path=./easy_train_data \
  --experiment-name=rtx4060_try1
````

若需要边训边测，把 --do-network-testing=True 并设：
- --network-testing-threads=4~8
- --network-testing-nodes-per-move=10000
- 以免拖慢训练。

常见问题
- Cupy 必须与 PyTorch/CUDA 版本匹配；RTX 4060 通常需 CUDA 12.x 生态。
- 数据/编译下载走 GitHub，国内网络建议提前准备镜像或手动缓存。
- gpus 参数在 easy_train.py 只用于决定要启多少个 run，每个 run 都会把单个 GPU ID 传给 train.py；不要把所有 ID 都直接传给 train.py。

需要我根据你机器的 CPU/磁盘具体规格，再帮你微调 num-workers/测试并发吗？