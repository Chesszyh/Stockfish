# Stats (统计)

`gather_statistics` 命令允许从 `.bin` 或 `.binpack` 文件中收集各种统计信息。语法为 `gather_statistics (GROUP)* input_file FILENAME`。可以指定多个组。任何属于至少一个指定组的统计收集器都将被使用。

最简单的用法：`stockfish.exe gather_statistics all input_file a.binpack`

任何未指定为参数名或不是参数的名称都将被解释为组名。

## 参数

`input_file` - 要读取的 `.bin` 或 `.binpack` 输入文件的路径。

`output_file` - (可选) 用于写入结果的输出文件路径。结果总是会写入控制台，因此如果指定此项，结果将同时写入两个地方。

`max_count` - 要处理的最大局面数。默认值：无限制。

## 组 (Groups)

`all` - 一个特殊的组，指定所有可用的统计信息收集器。

`position_count` - 文件中的总局面数。

`king`, `king_square_count` - 王在每个格子上的次数。输出以棋盘布局，第 8 横列在最上面。白王和黑王的值分开统计。

`move`, `move_from_count` - 与 `king_square_count` 相同，但用于 `from_sq(move)` (着法的起始格)。

`move`, `move_to_count` - 与 `king_square_count` 相同，但用于 `to_sq(move)` (着法的目标格)。

`move`, `move_type` - 每种类型着法的数量。包括普通、吃子、易位、升变、吃过路兵。这些组不是互斥的。

`move`, `moved_piece_type` - 每种类型的棋子被移动的次数。

`piece_count` - 棋盘上棋子数量的直方图。

`ply_discontinuities` - 两个连续局面之间，回合数跳跃值不为 1 的次数。通常等于对局数。

`material_imbalance` - 子力不平衡的直方图，使用“简单评估”计算值，即兵=1，象=马=3，车=5，后=9。

`results` - 对局结果的分布。

`endgames_6man` - 小于等于 6 个棋子（包括王）的残局配置分布。
