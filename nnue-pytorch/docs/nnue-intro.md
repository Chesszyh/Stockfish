
# **Stockfish-NNUE 深度解析：架构、训练与量化实现**

## **1\. NNUE 范式：国际象棋评估函数的一场革命**

国际象棋引擎的发展史是一部不断追求更深、更快、更准的搜索与评估的历史。在很长一段时间里，引擎的评估能力主要依赖于由人类专家精心设计的“手工评估函数”。然而，自 2020 年以来，一种名为 NNUE 的新范式彻底改变了这一格局，并推动 Stockfish 达到了前所未有的高度。

### **1.1. NNUE 之前的时代：经典评估的局限性**

在 NNUE 出现之前，顶级的 Alpha-Beta 搜索引擎（如早期版本的 Stockfish）使用一种复杂的、手工制作的评估函数 1。这个函数由成百上千个特征组成，每个特征都代表一个特定的国际象棋概念，例如：

* **子力平衡**：最基本的评估，计算双方棋子的价值总和。  
* **棋子-位置表（Piece-Square Tables, PSTs）**：为每个棋子在棋盘上的不同位置赋予不同的价值。例如，位于中心的马通常比位于棋盘边缘的马更有价值。  
* **兵形结构**：评估叠兵、孤兵、通路兵等兵形结构的优劣。  
* **王安全性**：评估王周围的保护情况。

这些特征的权重由国际象棋专家和引擎开发者凭经验设定，并通过一个名为 Fishtest 的大规模分布式测试框架进行微调 1。Fishtest 允许志愿者贡献 CPU 时间，通过运行数以百万计的对局来验证一个微小的改动是否能带来统计上显著的 Elo 等级分提升。

尽管这种方法取得了巨大成功，但其局限性也日益凸显。首先，手工设计的特征很难捕捉到棋局中高度复杂的、非线性的位置关系。许多深奥的战略思想，例如空间优势、子力协调或长期的局面压力，难以被量化为简单的加权和。其次，调整这些特征的权重是一项极其耗时且依赖专家知识的工作，其改进速度已逐渐趋于饱和。

### **1.2. NNUE 的突破：核心原理**

NNUE，全称为“高效可更新神经网络”（Efficiently Updatable Neural Network），最初是为日本将棋（Shogi）设计的，其设计理念旨在克服传统神经网络在 Alpha-Beta 搜索框架中应用的主要障碍——评估速度 2。NNUE 的成功建立在三个紧密相连的核心原理之上，使其能够在 CPU 上实现惊人的评估速度。

1. **高效可更新性 (Efficient Updatability)**：这是 NNUE 最具革命性的特点。在 Alpha-Beta 搜索中，引擎需要评估数百万个位置，而相邻的两个位置之间通常只有一个棋子发生了移动。传统的深度神经网络需要对每个新位置从头开始进行完整的正向传播计算，这对于 CPU 来说过于缓慢。NNUE 的架构设计允许在棋子移动后，通过增量更新（Incremental Update）的方式来计算新的评估值，而不是完全重新计算 3。这种更新操作在计算上极为廉价，从而使得评估速度能够跟上搜索的需求。  
2. **输入稀疏性 (Input Sparsity)**：国际象棋的局面本质上是稀疏的。在一个合法的棋局中，最多只有 32 个棋子占据着 64 个格子中的一部分。一个棋子的移动，通常只会改变棋盘上少数几个格子的状态。NNUE 的输入特征表示法巧妙地利用了这一点。尽管总特征空间可能非常巨大（例如，数十万维），但在任何给定局面下，只有极少数特征是“激活”的（即非零值）7。这种稀疏性使得网络的第一层（也是最大的一层）的计算可以被极大地简化和加速。  
3. **面向 CPU 的设计 (CPU-Centric Design)**：与 Leela Chess Zero 等依赖 GPU 进行大规模并行计算的引擎不同，NNUE 从一开始就是为在标准 CPU 上高效运行而设计的 3。其网络结构相对简单且“浅”（通常只有 3-4 个全连接层），这使得它非常适合采用低精度整数运算。通过利用现代 CPU 上的 SIMD（单指令多数据流）指令集（如 AVX2, AVX-512, VNNI），NNUE 可以在一个时钟周期内完成多个整数运算，从而实现极高的推理吞吐量，而无需昂贵的 GPU 硬件支持 3。

### **1.3. 对 Stockfish 的影响：新王者的诞生**

2020 年 8 月，NNUE 被正式合并到 Stockfish 的主分支中，催生了 Stockfish 12 1。这一变革带来了立竿见影的、颠覆性的效果。Fishtest 的测试结果显示，尽管 NNUE 版本的 Stockfish 每秒搜索的节点数（NPS）相比传统版本大幅下降（大约减半），但其棋力却获得了约 80-100 Elo 等级分的惊人提升 2。这表明 NNUE 评估的“质量”远高于传统评估，即使在搜索节点更少的情况下，也能做出更准确的判断。

最初，Stockfish 采用了一种混合评估系统：在子力相对平衡的局面中使用 NNUE，而在子力严重失衡的局面（例如，残局）中则回退到经典的、手工制作的评估函数 3。这种混合策略在当时被证明是有效的，可以在利用 NNUE 强大位置理解能力的同时，避免其在某些极端情况下的潜在弱点。然而，随着 NNUE 架构和训练方法的不断成熟，其评估能力变得越来越全面和可靠。最终，在 2023 年发布的 Stockfish 16 中，手工评估函数被完全移除，标志着 Stockfish 彻底过渡到了一个纯粹由神经网络驱动评估的时代 1。

### **1.4. 范式变革的深层逻辑**

Stockfish-NNUE 的成功不仅仅是一次技术升级，更代表了国际象棋引擎设计理念的一次深刻变革。它开创了一条介于传统引擎和 AlphaZero 风格引擎之间的“第三条道路”。

传统的 Alpha-Beta 引擎将强大的搜索算法与人类知识的结晶（手工评估）相结合。而以 AlphaZero 和 Leela Chess Zero 为代表的引擎则采用了蒙特卡洛树搜索（MCTS）与在 GPU 上运行的深度卷积神经网络（CNN）相结合的范式，几乎完全摒弃了人类先验知识 3。NNUE 的巧妙之处在于，它成功地将机器学习的强大威力嫁接到了久经考验的 Alpha-Beta 搜索框架上。它没有盲目地追求更深、更复杂的网络，而是设计了一个专门为满足 Alpha-Beta 搜索对低延迟、高吞吐量、CPU 原生评估的需求而量身定制的“浅层”网络。这种融合被证明比之前两种范式都更为强大和高效 1。

此外，NNUE 的架构设计是“性能驱动的架构妥协”的典范。在主流的深度学习领域，研究者们通常追求更优的泛化能力和更高的模型精度，可能会采用更复杂的激活函数（如 GeLU）或更深的网络结构。然而，NNUE 的设计者们做出了务实的选择，一切为最终部署环境下的推理速度服务。例如，网络中广泛使用的 ClippedReLU（将激活值限制在 0 和 1 之间）激活函数，其主要目的并非为了获得更好的学习动态，而是为了简化后续的量化过程 5。量化是将模型从浮点数转换为低精度整数的关键步骤，而一个有界的激活值范围使得这种转换的误差更容易控制。这清晰地表明，最终的部署目标——一个在 C++ 引擎中以惊人速度运行的评估函数——深刻地影响和决定了上游 Python 训练框架中的模型架构选择。

## **2\. Stockfish-NNUE 架构剖析**

Stockfish-NNUE 的核心是一个看似简单但设计精巧的全连接神经网络。其标准架构（以 halfkp\_256x2-32-32 为例）虽然只有几层，但每一层都为实现高效更新和快速 CPU 推理而精心设计。

### **2.1. 高层网络拓扑**

一个典型的 Stockfish-NNUE 网络由以下几个部分组成，形成一个从稀疏输入到单一评估值的计算流 3：

* **第 0 层 (特征变换器, Feature Transformer)**：这是网络的第一层，也是最大的一层。它本质上是一个巨大的、稀疏的线性层。其输入是代表棋盘局面的高维特征向量（对于 HalfKP 特征集，每个视角有 41,024 个特征）。该层将这些稀疏的输入特征映射到一个相对低维的、密集的向量空间。在 halfkp\_256x2-32-32 架构中，这个输出向量的维度是 256。由于输入是分两个视角（白方和黑方）提供的，所以这一层实际上会产生两个 256 维的向量。  
* **第 1 层 (隐藏层 1\)**：这一层的输入是将来自特征变换器的两个 256 维向量根据执子方（Side to Move）顺序拼接而成的 512 维向量。该向量首先经过一个 ClippedReLU 激活函数，然后通过一个全连接层，将其从 512 维映射到 32 维。  
* **第 2 层 (隐藏层 2\)**：与第一隐藏层类似，这一层也包含一个 ClippedReLU 激活函数和一个全连接层。它将输入的 32 维向量再次映射到 32 维。  
* **第 3 层 (输出层)**：这是网络的最后一层，是一个简单的线性层，它将来自第二隐藏层的 32 维向量映射为一个单一的标量值。这个标量值就是最终的、未经缩放的局面评估分数。

### **2.2. 累加器：高效更新的心脏**

“累加器”（Accumulator）是理解 NNUE 高效更新机制的关键概念。它并不是一个独立的网络层，而是指**特征变换器（第一层）在线性计算之后、应用激活函数之前**的输出结果 6。这个中间结果被存储在一个特殊的数据结构中。

累加器的核心作用在于：当棋盘上发生一步棋时，引擎无需重新计算整个特征变换器的矩阵乘法。由于输入特征的改变是局部的（通常只有几个特征从 0 变为 1，几个从 1 变为 0），引擎可以执行以下高效的增量更新操作 5：

1. 从当前的累加器值中，**减去**那些因棋子移动而“失效”的特征所对应的权重列。  
2. 在结果上，**加上**那些因棋子移动而“新激活”的特征所对应的权重列。

这个更新过程只涉及加法和减法，计算成本极低。完成累加器的更新后，引擎才需要对这个更新后的（相对较小的）密集向量应用 ClippedReLU 激活函数，并进行后续隐藏层的完整前向传播。由于后续层非常小，这个计算同样非常迅速。因此，整个评估过程被巧妙地分为了一个廉价的增量更新部分和一个快速的完整计算部分，从而实现了惊人的整体性能 3。

### **2.3. 双重视角与学习步调**

为了让网络能够理解同一个棋子布局在不同执子方下的价值差异，NNUE 采用了双重视角（Dual Perspectives）的设计。网络会为白方和黑方分别维护一个独立的累加器，我们称之为 Aw​ 和 Ab​ 5。

这两个累加器（均为 256 维向量）随后会被拼接成一个 512 维的向量，作为下一层的输入。这里的关键技巧在于拼接的顺序取决于当前轮到哪一方走棋（Side to Move, STM）5：

* 如果当前是**白方**走棋，拼接后的向量为 \[Aw​,Ab​\]。  
* 如果当前是**黑方**走棋，拼接后的向量为 \[Ab​,Aw​\]。

这个简单而绝妙的设计，将“步调”（Tempo）或“主动权”这个至关重要的国际象棋概念，以一种明确的方式编码到了网络的输入中。它使得网络能够学习到，完全相同的棋子物理位置，其评估价值会因为执子方的不同而产生显著变化。

### **2.4. 代码分析：model.py 中的 NNUE 模块**

在官方的 nnue-pytorch 训练代码库中，上述架构被实现在 model.py 文件的一个 torch.nn.Module 子类中。以下是其核心逻辑的简化代码示例，清晰地展示了数据流 11：

Python

\# 基于 \[11\] 的代码片段，展示了核心的前向传播逻辑  
import torch  
import torch.nn as nn

\# 假设的维度常量  
NUM\_FEATURES \= 41024 \* 2 \# 简化为总输入维度  
M \= 256 \# 第一个隐藏层（累加器）的维度  
N \= 32  \# 第二个隐藏层的维度  
K \= 32  \# 第三个隐藏层的维度

class NNUE(nn.Module):  
    def \_\_init\_\_(self):  
        super(NNUE, self).\_\_init\_\_()  
        \# 特征变换器，输入为白方和黑方特征的总和  
        self.input \= nn.Linear(NUM\_FEATURES, 2 \* M)  
        \# 后续隐藏层  
        self.l1 \= nn.Linear(2 \* M, N)  
        self.l2 \= nn.Linear(N, K)  
        self.output \= nn.Linear(K, 1)

    def forward(self, features, stm):  
        \# features 是一个稀疏的 (batch\_size, NUM\_FEATURES) 张量  
        \# stm 是一个 (batch\_size, 1\) 张量，白方走棋为 1.0，黑方为 0.0

        \# 1\. 特征变换器  
        \# 在实际实现中，这里会使用一个特殊的稀疏层来处理白方和黑方特征  
        \# 并生成两个 M 维的累加器 w 和 b  
        \# 此处为概念简化：  
        \# w \= self.feature\_transformer(white\_features)  
        \# b \= self.feature\_transformer(black\_features)  
          
        \# 假设我们已经通过某种方式得到了 w 和 b  
        \# w, b 的形状均为 (batch\_size, M)  
          
        \# 为了演示拼接逻辑，我们假设 self.input 直接产生了拼接好的 w 和 b  
        \# 实际代码中，输入是分开的 white\_features 和 black\_features  
        \# w, b \= torch.chunk(self.input(features), 2, dim=1) \# 概念性代码  
          
        \# 假设 w 和 b 已经获得  
        w \= torch.randn(features.shape, M)  
        b \= torch.randn(features.shape, M)

        \# 2\. 根据执子方（stm）拼接累加器  
        \# stm 乘以一个顺序，(1-stm) 乘以另一个顺序，然后相加  
        \# 这是一个很巧妙的、无需 if-else 的批量操作方式  
        accumulator\_cat \= (stm \* torch.cat(\[w, b\], dim=1)) \+ \\  
                          ((1.0 \- stm) \* torch.cat(\[b, w\], dim=1))

        \# 3\. 通过隐藏层和 ClippedReLU 激活函数  
        \# torch.clamp(x, min, max) 完美实现了 ClippedReLU  
        l1\_out \= torch.clamp(self.l1(accumulator\_cat), 0.0, 1.0)  
        l2\_out \= torch.clamp(self.l2(l1\_out), 0.0, 1.0)  
          
        \# 4\. 输出层  
        return self.output(l2\_out)

这段代码清晰地展示了双重视角拼接的核心思想。通过 stm 和 (1 \- stm) 的乘法，可以高效地为批次中的每个样本选择正确的拼接顺序，而无需使用条件判断，这对于在 GPU 上进行并行化训练至关重要。

## **3\. 深度解析：HalfKP 特征集编码**

NNUE 的强大性能不仅源于其高效的架构，更源于其精妙的输入特征工程。在众多特征集中，HalfKP（Half-King-Piece）是 Stockfish-NNUE 最经典、最成功的特征集之一。理解 HalfKP 的编码原理，是理解 NNUE 如何“看待”棋盘的关键。

### **3.1. “王是万物中心”的哲学**

HalfKP 的核心设计哲学是：棋盘上任何一个棋子的价值和作用，都与己方王的位置密切相关 3。它没有采用简单的

(棋子, 位置) 特征，而是使用了一种更复杂的、三元组式的特征：(王的位置, 棋子的位置, 棋子的类型)。

这意味着，对于每一个可能的王的位置（共 64 个），网络都存在一套独立的输入参数。可以将其想象成，NNUE 内部实际上包含了 64 个针对不同王位置的“子网络”。例如，当白王在 g1 时激活的参数，与白王在 c1 时激活的参数是完全不同的。这种设计将王在国际象棋中的核心地位——攻防的焦点、战术的中心——以一种非常底层和强大的方式根植于网络的结构之中。

“Half”一词则指代这种编码方式是从黑白双方各自的视角（Perspective）独立进行的。即，为白方生成一套以白王为中心的特征，同时为黑方生成另一套以黑王为中心的特征 3。

### **3.2. 构建 HalfKP 特征向量**

下面以白方的视角为例，逐步解析 HalfKP 特征向量的构建过程：

1. **确定王的位置**：首先，找到白王在棋盘上的位置，例如 e1 (方格索引为 4)。这个位置决定了当前将要使用 64 个“子网络”中的哪一个。  
2. **遍历其余棋子**：然后，遍历棋盘上除了双方王以外的所有棋子。  
3. **激活特征索引**：对于每一个被遍历到的棋子，例如在 d2 (方格索引为 11\) 上的一个白兵，系统会根据 (王的位置, 棋子的位置, 棋子类型) 这个三元组计算出一个唯一的特征索引。这个索引对应的输入神经元将被激活（设置为 1.0），而所有其他神经元保持为 0。

特征空间大小计算：  
HalfKP 特征集的总维度可以通过以下方式精确计算 3：

* **棋子类型**：除了王之外，双方各有 5 种棋子（兵、马、象、车、后）。因此，总共有 5×2=10 种需要编码的棋子类型。  
* **王的位置**：王可能在棋盘上的 64 个方格中的任意一个。  
* **棋子的位置**：其他棋子也可能在 64 个方格中的任意一个。  
* **零位棋子 (BONA\_PIECE\_ZERO)**：这是一个源自将棋的特殊特征，为每个王的位置额外增加一个特征维度。  
* 总维度（单方视角）：因此，对于单方（例如白方）来说，总的特征维度是：  
  64 (王的位置)×(64 (棋子位置)×10 (棋子类型)+1 (零位棋子))=64×(640+1)=41,024  
* **总输入维度**：由于需要为黑白双方分别编码，所以网络的总输入特征维度为 41,024×2=82,048。

### **3.3. HalfKP 棋子类型与索引逻辑**

为了将上述概念与实际代码联系起来，理解棋子类型如何映射到具体的索引至关重要。在 nnue-pytorch 的实现中，10 种棋子类型被赋予了从 0 到 9 的整数索引。

| 棋子 | 颜色 | 代码中的 piece\_type 索引 |
| :---- | :---- | :---- |
| 兵 (Pawn) | 白 | 0 |
| 兵 (Pawn) | 黑 | 1 |
| 马 (Knight) | 白 | 2 |
| 马 (Knight) | 黑 | 3 |
| 象 (Bishop) | 白 | 4 |
| 象 (Bishop) | 黑 | 5 |
| 车 (Rook) | 白 | 6 |
| 车 (Rook) | 黑 | 7 |
| 后 (Queen) | 白 | 8 |
| 后 (Queen) | 黑 | 9 |

基于此，一个特征的最终索引可以（概念上）通过以下公式计算：  
Index=king\_sq×(64×10+1)+piece\_sq×10+piece\_type\_index  
其中 king\_sq 和 piece\_sq 是棋盘方格的索引（0-63）。这个公式清晰地展示了特征索引是如何由王的位置、棋子位置和棋子类型唯一确定的。

### **3.4. 代码分析：halfkp.py 和 features.py**

在 nnue-pytorch 代码库中，halfkp.py 和 features.py 文件共同实现了 HalfKP 的编码逻辑。

halfkp.py 中的 halfkp\_idx 函数是计算特征索引的核心 13。它接收王的位置、棋子位置和棋子对象作为参数，并返回最终的索引值。该文件还包含一个重要的

orient 函数：orient(is\_white\_pov, sq) \= (63 \* (not is\_white\_pov)) ^ sq。这个函数用于处理不同视角的坐标转换。

* 当从白方视角（is\_white\_pov 为 True）观察时，orient 函数直接返回原始的方格索引 sq。  
* 当从黑方视角（is\_white\_pov 为 False）观察时，函数执行 63 ^ sq 操作。这是一个按位异或操作，相当于将棋盘旋转 180 度。这是一个源自将棋的习惯，将棋棋盘具有 180 度旋转对称性。这与更直观的垂直翻转（通常是 56 ^ sq）有所不同，但已被证明同样有效 3。

features.py 中的 Features 类（特别是其 get\_active\_features 方法）则负责调用 halfkp\_idx 来为整个棋盘生成激活的特征列表 13。该方法会遍历棋盘上的所有棋子（除了王），为每个棋子计算出对应的 HalfKP 特征索引，并在一个稀疏张量（Tensor）的相应位置上标记为 1.0。

### **3.5. 通过因子分解 (HalfKP^) 提升泛化能力**

尽管 HalfKP 非常强大，但其严格的 (王, 棋子) 绑定关系也带来了一个潜在的缺点：泛化能力可能受限。例如，网络可能很难学习到一个普遍的规则，如“车在开放线上是好的”，因为这个规则在 64 个不同的王的位置下，对应的是 64 组完全不同的权重 14。

为了解决这个问题，研究者们引入了“因子分解特征集”（Factorized Features），在命名上通常用 ^ 符号表示，例如 HalfKP^ 15。

其核心思想是在**训练期间**引入一些额外的、更简单的“虚拟特征”。这些虚拟特征不依赖于王的位置。例如，可以引入：

* **P 特征**：简单的 (棋子, 位置) 特征。  
* **K 特征**：仅与王的位置相关的特征。

在训练时，网络同时学习主 HalfKP 特征和这些虚拟特征的权重。当训练完成，需要将模型序列化为最终的 .nnue 文件时，这些虚拟特征的权重会被“合并”（Coalesced）或“吸收”回主 HalfKP 特征的权重中。具体来说，一个 HalfKP 特征 (k, p, t) 的最终权重，会是其自身学习到的权重，加上对应的 P 特征 (p, t) 的权重，再加上对应的 K 特征 (k) 的权重。

这种方法巧妙地结合了两种模式的优点：在训练时，网络可以利用简单的虚拟特征更容易地学习到普适的规则；在推理时，所有的知识都被烘焙（bake）进了单一、高效的 HalfKP 结构中，不会增加任何额外的计算开销 14。

## **4\. nnue-pytorch 训练流程**

一个强大的 NNUE 网络不仅需要精巧的架构，更依赖于一个完整、高效且经过精心设计的训练流程。nnue-pytorch 项目 17 提供了一整套从数据生成到模型训练和验证的工具链。

### **4.1. 数据管理的至要作用**

高质量的训练数据是训练成功的基石。NNUE 的数据处理流程有几个显著特点。

数据生成与格式：  
训练数据通常是通过让 Stockfish 引擎进行自对弈或分析大量局面来生成的 18。对于每个局面，会记录其 FEN（Forsyth-Edwards Notation）字符串以及引擎在特定深度下的评估分。这些数据被存储在一种高度优化的二进制格式中，以减小存储体积并加快读取速度。

* **.bin 格式**：这是早期的格式，每个训练样本（局面、评估分、游戏结果等）占用 40 字节 19。  
* **.binpack 格式**：这是一种更先进的压缩格式。它利用了自对弈游戏中局面之间的连续性，通过存储局面间的差异而非完整局面，将每个样本的平均大小压缩到了惊人的 2-3 字节 19。

为了在训练时达到最高的 I/O 性能，nnue-pytorch 使用了一个用 C++ 编写的高速数据加载器来解析这些二进制文件，而不是使用纯 Python 17。

为“宁静”局面进行筛选：  
这是 NNUE 训练流程中一个至关重要且不那么直观的步骤。社区的实践和研究发现，直接在所有生成的局面上进行训练，效果并不理想。最强大的网络是在所谓的“宁静局面”（Quiet Positions）上训练出来的 20。  
宁静局面指的是那些没有即时战术（如将军、吃子、捉双等）的、相对稳定的局面。筛选宁静局面的核心逻辑在于，NNUE 本质上是一个**静态评估函数**，其目标是给出一个稳定的、不依赖于短期搜索的局面价值判断。如果在充满战术变化的“嘈杂”局面上进行训练，数据标签（来自浅层搜索的评估分）会包含大量短期的、不稳定的战术评估，这会“污染”训练数据，导致网络学习到错误的或不稳定的知识 20。

数据来源：  
值得注意的是，许多用于训练官方 Stockfish 网络的顶级数据集，其原始数据来源于 Leela Chess Zero 项目的自对弈数据 23。这些海量数据经过社区的精心筛选、转换和处理，形成了用于 NNUE 训练的高质量语料库。

### **4.2. 训练循环：损失函数与优化器**

nnue-pytorch 代码库中的 train.py 脚本是整个训练过程的核心控制器 17。它实现了标准的 PyTorch 训练循环，但其损失函数的设计尤为独特。

混合损失函数：  
NNUE 的训练目标并非简单的回归或分类。其损失函数是一个混合体，由两个部分加权组成，权重由一个超参数 λ 控制 18：

1. **评估分损失 (MSE)**：这是训练的回归部分。它计算网络输出的评估分与训练数据中标注的“真实”评估分之间的均方误差（Mean Squared Error）。  
2. **对局结果损失 (WDL)**：这是训练的分类部分。网络输出的评估分会被通过一个 Sigmoid 函数映射到一个代表胜/平/负（Win/Draw/Loss）概率的值。然后，这个概率会与训练数据中记录的实际对局结果进行比较，通常使用交叉熵（Cross-Entropy）损失。

总损失可以表示为：  
Ltotal​=λ⋅LMSE​+(1−λ)⋅LWDL​  
λ 参数：  
这个参数控制着训练的侧重点。当 λ=1.0 时，训练完全忽略对局结果，只专注于拟合教师引擎的评估分（纯回归）。当 λ=0.0 时，训练则完全忽略评估分，只专注于预测对局的最终胜负（纯分类）。在实践中，通过调整 λ 值（例如设为 0.8 或 0.9），可以在拟合局面细节和学习长期胜负趋势之间找到一个最佳平衡点 18。  
缩放因子：  
在计算 MSE 损失之前，以“百分兵”（centipawn）为单位的评估分需要被缩放。这个缩放因子对于平衡来自 MSE 和 WDL 两部分的梯度大小至关重要。最初，社区使用了一个经验值 600，但后来的分析发现，这个值实际上是对一个基于兵的价值（PawnValue）的更复杂公式的近似，一个更精确的值大约在 361 左右 24。  
优化器：  
训练过程通常使用 Adam 优化器，这是一种在深度学习领域被广泛应用的、稳健且高效的自适应学习率优化算法 25。

### **4.3. 训练目标的深层解读**

NNUE 的训练方法揭示了其在 Stockfish 引擎中的真正角色。一个关键的观察是，训练数据中的评估分标签通常来自一个中等深度的 Stockfish 搜索（例如，深度 5 或 9），并且训练在非常深的搜索（如深度 20）生成的数据上效果反而很差 24。

这背后隐藏着一个深刻的设计理念：NNUE 的训练目标并**不是**像 AlphaZero 那样从零开始学习国际象棋的“绝对真理”，而是学习**快速地、准确地近似一个传统 Stockfish 搜索在浅层会得出的评估结论**。

换句话说，NNUE 被训练成了其“老师”（即传统评估函数+浅层搜索）的一个高速代理。它被设计用来完美地替代 Alpha-Beta 搜索树的前几层。当引擎需要评估一个新节点时，它可以立即调用 NNUE 获得一个高质量的静态评估，而无需再进行耗时的浅层搜索。这使得引擎能够以惊人的速度“剪枝”，即放弃那些明显不好的分支，从而将宝贵的计算时间集中在更有希望的变化上。这种“与搜索协同”而非“取代搜索”的哲学，是 NNUE 能够无缝集成到 Alpha-Beta 框架并带来巨大成功的根本原因。

## **5\. 深度解析：为高速推理而生的 NNUE 量化**

如果说精巧的架构和训练是 NNUE 强大棋力的“灵魂”，那么高效的量化则是其惊人速度的“肉体”。将在 PyTorch 中以 float32 精度训练好的模型，转换为 Stockfish C++ 引擎中可以高速运行的低精度整数模型，是整个流程中至关重要的一步。

### **5.1. 速度的需求：从 float32 到 int8**

一个标准的、使用 32 位浮点数（float32）的神经网络模型，对于国际象棋引擎每秒数百万次评估的需求来说，速度太慢了 5。量化（Quantization）是将模型的浮点数权重、偏置和激活值转换为低精度整数（主要是

int16 和 int8）的过程 5。

量化带来了两大核心优势：

1. **减小模型体积**：整数类型比浮点数类型占用更少的存储空间。一个量化后的 .nnue 文件通常只有几十兆字节，便于分发和加载 22。  
2. **加速计算**：这是最关键的好处。现代 CPU 提供了高度优化的 SIMD 指令集，能够在一个时钟周期内并行处理多个整数运算（例如，一条 AVX2 指令可以同时处理 32 个 int8 数据）。这使得整数矩阵乘法的速度远超浮点数计算，从而带来数量级的性能提升 3。

### **5.2. Stockfish 的量化方案**

Stockfish 采用的不是 PyTorch 等框架提供的标准量化工具，而是一套定制的、相当“激进”的量化方案，专为 NNUE 架构和 C++ 推理引擎优化。

* **特征变换器 (第 0 层)**：这一层的权重和偏置项从 float32 被量化为 int16 3。因此，累加器中存储的是  
  int16 类型的整数。  
* **隐藏层与输出层 (第 1, 2, 3 层)**：这些更小的层的权重和偏置项则被更激进地量化为 int8 3。  
* **激活值**：激活值的量化是整个方案的精髓。ClippedReLU 函数将激活函数的输出严格限制在 \[0.0, 1.0\] 的浮点数范围内。这个有界的范围随后被线性地映射到 int8 的整数范围 \`\` 3。这种设计有效地防止了量化误差在网络层间传递时被累积和放大，保证了模型的精度。

### **5.3. 量化的数学原理**

量化的核心是线性映射。一个浮点数值 v\_float 可以通过以下公式转换为一个量化后的整数值 v\_quant：  
vquant​=round(vfloat​×scale\_factor)  
在 Stockfish 的 C++ 推理代码中，这个过程是通过整数乘法和位移（代表除法）来实现的，serialize.py 脚本必须精确地复现这一过程。

* **特征变换器输出**：累加器中的 int16 值在作为下一层输入前，会被裁剪（clip）到 \`\` 的范围内，然后直接转换为 int8 类型。这个范围由 C++ 代码中的 kActivationMin 和 kActivationMax 常量定义。  
* **隐藏层计算**：在隐藏层中，int8 的输入激活值与 int8 的权重进行矩阵乘法，其结果会累加到一个 int32 的中间变量中。这个 int32 的和在经过激活函数前，需要被缩放。一个典型的缩放操作是右移 6 位，相当于除以 26=64 3。

  output=sum≫6  
* **最终输出**：输出层的 int32 结果同样需要被缩放，以使其符合引擎期望的“百分兵”（centipawn）单位。这个缩放因子由 FV\_SCALE 常量定义，通常是 16（即右移 4 位）3。

此外，社区还引入了 QA 和 QB 两个量化常数，它们作为缩放因子，允许在模型的精度（影响棋力）和整数值的范围（影响压缩率和潜在的溢出风险）之间进行权衡 22。

### **5.4. 各层量化参数总览**

下表总结了 Stockfish-NNUE 网络中每一层的量化参数，为理解整个数据流提供了一个清晰的参考。

| 网络层名称 | 权重类型 | 偏置类型 | 输入激活值类型 | 输出激活值类型 | 激活后缩放 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **特征变换器** | int16 | int16 | bool (稀疏 1.0) | int16 (累加器) | 不适用 (输出被裁剪为 int8) |
| **隐藏层 1 (FC1)** | int8 | int32 | int8 \`\` | int32 (缩放前) | 右移 6位 (\>\> 6\) |
| **隐藏层 2 (FC2)** | int8 | int32 | int8 \`\` | int32 (缩放前) | 右移 6位 (\>\> 6\) |
| **输出层** | int8 | int32 | int8 \`\` | int32 (最终分数) | 右移 4位 (\>\> 4, FV\_SCALE) |

### **5.5. 代码分析：serialize.py 脚本**

serialize.py 脚本是连接 PyTorch 训练环境和 C++ 推理引擎的桥梁。它的核心任务是将一个训练好的 PyTorch 模型检查点（.ckpt 或 .pt 文件）转换为 Stockfish 引擎可以直接加载和使用的、二进制的 .nnue 文件 15。

其工作流程大致如下：

1. **加载模型**：从检查点文件中加载使用 float32 存储的、训练好的模型权重和偏置。  
2. **权重合并**：如果训练时使用了因子分解特征（如 HalfKP^），脚本会执行权重合并操作，将虚拟特征的权重加到主特征变换器的相应权重上，从而得到一个单一的、完整的权重矩阵 16。  
3. **量化**：脚本应用上述的定制化量化逻辑。它遍历模型的每一层，将 float32 的权重和偏置张量，乘以相应的缩放因子并四舍五入，转换为 int16 和 int8 的整数张量。  
4. **打包与写入**：最后，脚本将所有量化后的整数张量按照 C++ 引擎期望的特定二进制布局（包括网络架构信息、权重和偏置数据等）打包，并写入到最终的 .nnue 文件中。

这个过程完成后，生成的 .nnue 文件就成了一个完全独立的、可被 Stockfish 直接使用的评估模块。

## **6\. 从网络文件到引擎决策：C++ 中的推理过程**

一个 .nnue 文件被创建出来后，它的生命周期才刚刚开始。它将被加载到 Stockfish 引擎中，在每一次搜索中被调用数百万次，成为引擎决策的核心依据。

### **6.1. 加载与探测网络**

当 Stockfish 引擎启动时，或者当用户通过 UCI（Universal Chess Interface）命令 setoption name EvalFile value /path/to/net.nnue 指定网络文件时，引擎会读取这个 .nnue 文件 2。文件中的二进制数据被解析，量化后的权重和偏置被加载到内存中，填充到 C++ 代码中预先定义好的数据结构里。从此，这个神经网络就准备好为引擎提供评估服务了。

### **6.2. evaluate.cpp 中的高性能评估**

src/evaluate.cpp 文件中的 Eval::evaluate 函数是 Stockfish 评估函数的对外接口，也是引擎搜索算法中的一个计算热点（hot path）28。当需要对一个局面进行完整评估时（例如，搜索的根节点，或者发生了一次无法增量更新的移动如王车易位），就会调用这个函数。

在这个函数内部，真正的魔法发生在针对特定 CPU 架构优化的底层函数中。这些函数大量使用 SIMD 内在函数（intrinsics），例如在支持 AVX2 的 CPU 上，会使用 \_mm256\_maddubs\_epi16 这样的指令来高效地执行 int8 矩阵乘法和累加操作 3。正是在这里，量化带来的理论性能优势被完全转化为实际的计算速度。

### **6.3. 增量更新的实现：Position::do\_move**

然而，对于搜索过程中的绝大多数节点，引擎并不会调用完整的 Eval::evaluate 函数。当引擎在搜索树中向前走一步（通过 Position::do\_move 函数）时，它会触发一个更高效的增量更新流程。

这个流程与第 2.2 节中描述的累加器理论完全对应。C++ 代码会精确地识别出这一步棋导致了哪些 HalfKP 特征的增加和删除，然后直接在内存中的 C++ 累加器数据结构上执行相应的加法和减法操作 3。只有在累加器被更新之后，引擎才会对这个小得多的累加器向量进行后续（同样是 SIMD 加速的）隐藏层和输出层的计算。

这种两级评估机制——对非王翼移动进行极快的增量更新，仅在必要时（如王移动或新局面）进行完整刷新——是 NNUE 设计哲学的最终体现，也是 Stockfish 能够在保持极高搜索速度的同时，享受到神经网络强大评估能力的根本原因。

## **7\. 总结与未来展望**

Stockfish-NNUE 的实现是一次集精巧理论、务实工程与社区协作于一体的伟大成功。它不仅将 Stockfish 推向了不可撼动的棋力王座，也为高性能计算与人工智能的结合提供了一个经典的范例。

### **7.1. 关键设计原则的综合**

回顾整个系统，我们可以清晰地看到几条贯穿始终的核心设计原则：

* **Alpha-Beta 与机器学习的深度融合**：NNUE 没有试图用一个大而全的神经网络取代所有，而是设计了一个专门服务于 Alpha-Beta 搜索框架的评估工具。它开创了区别于传统引擎和 AlphaZero 引擎的“第三条道路”，并证明了其优越性。  
* **性能优先的务实主义**：从 HalfKP 特征集对王中心地位的强调，到 ClippedReLU 对量化的友好，再到最终的 int8 SIMD 推理，整个技术栈的每一个决策都深刻地受到了最终 CPU 推理性能需求的驱动。  
* **“与搜索协同”的训练哲学**：NNUE 的训练目标是快速、准确地模拟浅层搜索的结果，使其成为搜索算法的“加速器”和“导航仪”，而非一个独立的“思考者”。

这一系列设计原则形成了一条紧密的因果链：对增量更新的需求，催生了累加器和稀疏输入的架构；对 CPU 高速推理的需求，决定了网络的浅层结构、激活函数的选择和激进的量化方案；而对与 Alpha-Beta 搜索协同的需求，则塑造了其独特的训练数据生成和筛选策略。

### **7.2. NNUE 不断演进的前景**

Stockfish-NNUE 并非一个已经完成的静态项目，而是一个在 Fishtest 平台上由全球社区驱动、持续演进的生命体 1。新的网络架构和训练技术正被不断地提出和测试。

例如，随着 SIMD 实现的进一步优化，社区已经能够成功地将第一隐藏层的维度从 256x2 提升到 512x2 甚至 1024x2，在可接受的速度损失下换取了更强的棋力 29。同时，对更深或更复杂的层级结构（如

1024-\>16-\>32-\>1）的实验也在不断进行，以探索速度与强度之间的最佳平衡点 27。

可以预见，未来的 Stockfish-NNUE 将在特征工程（例如，探索超越 HalfKP 的新特征集）、网络架构（例如，引入更高效的层类型）和训练方法（例如，更先进的自对弈和数据筛选技术）等多个方面继续突破。它将继续作为 CPU 国际象棋评估技术的巅峰之作，不断刷新我们对机器智能极限的认知。

#### **Works cited**

1. Stockfish (chess) \- Wikipedia, accessed August 8, 2025, [https://en.wikipedia.org/wiki/Stockfish\_(chess)](https://en.wikipedia.org/wiki/Stockfish_\(chess\))  
2. Introducing NNUE Evaluation \- Stockfish \- Strong open-source chess engine, accessed August 8, 2025, [https://stockfishchess.org/blog/2020/introducing-nnue-evaluation/](https://stockfishchess.org/blog/2020/introducing-nnue-evaluation/)  
3. Stockfish NNUE \- Chessprogramming wiki, accessed August 8, 2025, [https://www.chessprogramming.org/Stockfish\_NNUE](https://www.chessprogramming.org/Stockfish_NNUE)  
4. Efficiently updatable neural network \- Wikipedia, accessed August 8, 2025, [https://en.wikipedia.org/wiki/Efficiently\_updatable\_neural\_network](https://en.wikipedia.org/wiki/Efficiently_updatable_neural_network)  
5. NNUE | Stockfish Docs \- GitHub Pages, accessed August 8, 2025, [https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)  
6. NNUE \- Chessprogramming wiki, accessed August 8, 2025, [https://www.chessprogramming.org/NNUE](https://www.chessprogramming.org/NNUE)  
7. variant-nnue-pytorch/docs/nnue.md at master \- GitHub, accessed August 8, 2025, [https://github.com/fairy-stockfish/variant-nnue-pytorch/blob/master/docs/nnue.md](https://github.com/fairy-stockfish/variant-nnue-pytorch/blob/master/docs/nnue.md)  
8. Stockfish NNUE \- Chessprogramming wiki, accessed August 8, 2025, [https://www.chessprogramming.org/index.php?title=Stockfish\_NNUE\&mobileaction=toggle\_view\_desktop](https://www.chessprogramming.org/index.php?title=Stockfish_NNUE&mobileaction=toggle_view_desktop)  
9. NNUE merge · Issue \#2823 · official-stockfish/Stockfish \- GitHub, accessed August 8, 2025, [https://github.com/official-stockfish/Stockfish/issues/2823](https://github.com/official-stockfish/Stockfish/issues/2823)  
10. A Theoretical Analysis of the Development and Design Principles of NNUE for Chess Evaluation \- International Journal of Research and Innovation in Applied Science (IJRIAS), accessed August 8, 2025, [https://rsisinternational.org/journals/ijrias/articles/a-theoretical-analysis-of-the-development-and-design-principles-of-nnue-for-chess-evaluation/](https://rsisinternational.org/journals/ijrias/articles/a-theoretical-analysis-of-the-development-and-design-principles-of-nnue-for-chess-evaluation/)  
11. How to understand 'perspective' in NNUE nets? \- TalkChess.com, accessed August 8, 2025, [https://talkchess.com/viewtopic.php?t=83988](https://talkchess.com/viewtopic.php?t=83988)  
12. NNUE eval rotate vs mirror · Issue \#3021 · official-stockfish/Stockfish \- GitHub, accessed August 8, 2025, [https://github.com/official-stockfish/Stockfish/issues/3021](https://github.com/official-stockfish/Stockfish/issues/3021)  
13. nnue-pytorch/halfkp.py at master \- GitHub, accessed August 8, 2025, [https://github.com/glinscott/nnue-pytorch/blob/master/halfkp.py](https://github.com/glinscott/nnue-pytorch/blob/master/halfkp.py)  
14. Pytorch NNUE training \- Page 4 \- TalkChess.com, accessed August 8, 2025, [https://talkchess.com/viewtopic.php?t=75724\&start=30](https://talkchess.com/viewtopic.php?t=75724&start=30)  
15. chess variant NNUE training code for Fairy-Stockfish \- GitHub, accessed August 8, 2025, [https://github.com/fairy-stockfish/variant-nnue-pytorch](https://github.com/fairy-stockfish/variant-nnue-pytorch)  
16. Features | Stockfish Docs \- GitHub Pages, accessed August 8, 2025, [https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/features.html](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/features.html)  
17. official-stockfish/nnue-pytorch: Stockfish NNUE (Chess ... \- GitHub, accessed August 8, 2025, [https://github.com/official-stockfish/nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch)  
18. joergoster/Stockfish-NNUE: UCI Chess engine Stockfish with an Efficiently Updatable Neural-Network-based evaluation function \- GitHub, accessed August 8, 2025, [https://github.com/joergoster/Stockfish-NNUE](https://github.com/joergoster/Stockfish-NNUE)  
19. Variant NNUE training data generator for Fairy-Stockfish \- GitHub, accessed August 8, 2025, [https://github.com/fairy-stockfish/variant-nnue-tools](https://github.com/fairy-stockfish/variant-nnue-tools)  
20. Study of the Proper NNUE Dataset \- arXiv, accessed August 8, 2025, [https://arxiv.org/html/2412.17948v1](https://arxiv.org/html/2412.17948v1)  
21. arXiv:2412.17948v1 \[cs.AI\] 23 Dec 2024, accessed August 8, 2025, [https://arxiv.org/pdf/2412.17948](https://arxiv.org/pdf/2412.17948)  
22. My solution: Cfish, nnue, data (1st) \- Kaggle, accessed August 8, 2025, [https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/writeups/linrock-my-solution-cfish-nnue-data-1st](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/writeups/linrock-my-solution-cfish-nnue-data-1st)  
23. Stockfish NNUE training data, accessed August 8, 2025, [https://robotmoon.com/nnue-training-data/](https://robotmoon.com/nnue-training-data/)  
24. Pytorch NNUE training \- Page 6 \- TalkChess.com, accessed August 8, 2025, [https://talkchess.com/viewtopic.php?t=75724\&start=50](https://talkchess.com/viewtopic.php?t=75724&start=50)  
25. Optimizing Model Parameters — PyTorch Tutorials 2.8.0+cu128 documentation, accessed August 8, 2025, [https://docs.pytorch.org/tutorials/beginner/basics/optimization\_tutorial.html](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)  
26. Use PyTorch to train your data analysis model | Microsoft Learn, accessed August 8, 2025, [https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-analysis-train-model](https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-analysis-train-model)  
27. Stockfish introduces a new NNUE network architecture and associated network parameters : r/chess \- Reddit, accessed August 8, 2025, [https://www.reddit.com/r/chess/comments/nfp1m4/stockfish\_introduces\_a\_new\_nnue\_network/](https://www.reddit.com/r/chess/comments/nfp1m4/stockfish_introduces_a_new_nnue_network/)  
28. Stockfish/src/evaluate.cpp at master \- GitHub, accessed August 8, 2025, [https://github.com/official-stockfish/Stockfish/blob/master/src/evaluate.cpp](https://github.com/official-stockfish/Stockfish/blob/master/src/evaluate.cpp)  
29. Stockfish Development Builds \- Next Chess Move, accessed August 8, 2025, [https://m.nextchessmove.com/dev-builds/d61d38586ee35fd4d93445eb547e4af27cc86e6b](https://m.nextchessmove.com/dev-builds/d61d38586ee35fd4d93445eb547e4af27cc86e6b)