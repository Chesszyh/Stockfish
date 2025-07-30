# Note

## Getting Started

### Structure

- `movegen.h` / `movegen.cpp`：生成所有合法着法

- `evaluate.h` / `evaluate.cpp`：

评估过程推结果是容易的，但解释结果可能是困难的。增加调试日志？

How to? 

评分加权的追溯？

`search.h` / `search.cpp`：

搜索算法剪枝的原因？“为什么某个看起来不错的走法，在深入计算后被引擎排除了”？

### 实践

- stdin/out跟踪uci进程
- 构建解释模型

## Evaluation

### Centipawns

1 pawn = 100 centipawns -> 50% winning chance

展示胜率：`UCI_ShowWDL=True`

### Tablebase Scores

200 pawns = 1 tablebase win

199.50 -> 25步之后达到标准胜局（50*0.5）

## Threads

Stockfish使用多线程来加速搜索，默认只开启单线程，可以设置为最大线程数-1。使用`lscpu`查看CPU信息，如果有：`Thread(s) per core:      2`，则表示每个核心有两个线程，当前CPU支持超线程。

我的情况，应该可以开到60.

## Hash

Stockfish使用哈希表来存储搜索过程中遇到的局面，以避免重复计算。默认情况下，哈希表大小为16MB。

## ELO rating

引擎从常规局面下开始对弈，大部分都将是和局，会导致等级分差距极小。Fishtest框架提供了相对公平的竞技条件。

## Training

Stockfish的NNUE架构适合在CPU上跑，但是nnue的训练是在GPU上。

## Minimax

Stockfish的minimax使用pruning, reductions(浅搜索低评分的分支，而不是直接完全忽略)，extensions(深搜索高评分的分支)，策略比较激进，可能导致有时候漏掉一些短将杀。

例如，Stockfish 15.1 在高深度时分支因子只有 1.5，也就是说在第 50 层只考虑了约 1.5 个走法，而不是全部 20-40 个可能走法。




