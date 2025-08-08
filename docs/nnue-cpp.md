# nnue-cpp

[nnue-cpp](https://github.com/joergoster/Stockfish-NNUE)：使用`stockfish-nnue`等工具

Reference: https://www.zhihu.com/people/wan-ma-46/posts 等五篇文章

## Bitboard

### 己方与对方

nnue使用Nega-max（对minimax的改进）:

```cpp
int NegaMax(int depth) {
    if (depth == 0) return evaluate(side_to_move);
    int max = -inf;
    for (all_moves) {
        score = -NegaMax(depth - 1);
        if (score > max)
            max = score;
    }
    return max;
}
```

默认白方为己方，黑方需要翻转棋盘。nnue是进行中心镜像翻转(上下+左右)，实际上只进行上下翻转可能更合适。

### 单方棋盘表示

使用**位置关系编码**，表示某一方的王与棋盘上每一个其他子力的位置关系。

41024维向量，其中41024 = 64（己方的王所在格子的枚举个数） * （2（己方+对方） * 5（棋子种类）* 64（棋盘格数） + 1（上一步是否为吃子着法）），整个（2*5*64 + 1）表示除对方王之外的棋子所在格子。

位置关系枚举值 = 己方王格子编号 * 641 + 棋子种类编号 * 64 + 棋子格子编号 + 1

nnue使用的41024维高维稀疏向量有利于增量更新，能够提高计算速度。

## Network Structure

## Training Dataset

NNUE的训练数据存储在一个二进制文件中，其中，每一个样本包含输入（已经提取特征了的feature vector或未经处理的原始数据）和标签。在NNUE的代码中，每个训练样本被称为`PackedSfenValue`。

### `PackedSfenValue`结构体

```cpp
// file: nnue-pytorch/lib/nnue_training_data_formats.h
struct PackedSfenValue
{
    // phase
    PackedSfen sfen;

    // Evaluation value returned from Learner::search()
    int16_t score;

    // PV first move
    // Used when finding the match rate with the teacher
    StockfishMove move;

    // Trouble of the phase from the initial phase.
    uint16_t gamePly;

    // 1 if the player on this side ultimately wins the game. -1 if you are losing.
    // 0 if a draw is reached.
    // The draw is in the teacher position generation command gensfen,
    // Only write if LEARN_GENSFEN_DRAW_RESULT is enabled.
    int8_t game_result;

    // When exchanging the file that wrote the teacher aspect with other people
    //Because this structure size is not fixed, pad it so that it is 40 bytes in any environment.
    uint8_t padding;

    // 32 + 2 + 2 + 2 + 1 + 1 = 40bytes
};
```

nnue学习目标，是`game_result`和`score`的线性组合。`score`来自`Learner::search()`，即固定深度的alpha-beta搜索。如果单纯依赖`score`，则无法避免手写估值函数自身缺陷对训练的负面影响：

如果这个手写函数的估值不准确，那么NNUE的训练就会被它误导，结果恐怕不会太好；

如果这个手写函数的估值非常准确，那NNUE的训练成果到底该归功于训练本身，还是这个手写函数提供的高质量数据？而且既然已经有了这么准确的手写估值函数，干嘛还要再训练一个NNUE网络呢？

### 自对弈策略

添加N步随机着法，避免过多重复样本。

- 纵向随机：给定N个随机着法配额的前提下，在对局中的哪几步添加随机着法？
    - Continuous Moves from Initial Position
    - Discontinuous Moves Given Min-ply and Max-ply
- 横向随机：在所有合法着法中选择哪一个着法作为“随机着法”？
    - Random Move from All (King) Legal Moves
    - Random Move from Top-k Legal Moves(Multi-PV Moves)

自对弈对局可以高度并行化处理，对局存储由`SfenWriter`消息对队列实现。

### 实践

```bash
# 检查机器支持的指令集，选择最新的可以提高性能
cat /proc/cpuinfo | grep flags | head -1

# 编译
make nnue ARCH=x86-64-bmi2
mv stockfish stockfish-nnue

make nnue-gen-sfen-from-original-eval ARCH=x86-64-bmi2
mv stockfish stockfish-nnue-gen-sfen

make nnue-learn ARCH=x86-64-bmi2
mv stockfish stockfish-nnue-learn
```

# nnue-pytorch

[nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch/)：使用Docker
