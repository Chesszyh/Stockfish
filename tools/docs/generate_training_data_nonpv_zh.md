# generate_training_data_nonpv (非主变训练数据生成)

`generate_training_data_nonpv` 命令允许通过自对弈生成训练数据，其方式比传统对局更适合训练。它通过固定节点数的自对弈进行探索，并记录部分评估过的局面，然后使用固定深度的搜索对这些局面进行重新评分。

与 Stockfish 中的所有命令一样，`generate_training_data_nonpv` 既可以从命令行调用（例如 `stockfish.exe generate_training_data_nonpv ...`，但不推荐这样做，因为在命令执行前无法设置 UCI 选项），也可以在交互式提示符中调用。

建议将 `PruneAtShallowDepth` UCI 选项设置为 `false`，这将提高固定深度搜索的质量。

建议将 `EnableTranspositionTable` UCI 选项保持为默认值 `true`，这会加快生成过程，且不会显著损害数据的均匀性。

`generate_training_data_nonpv` 命令接受命名参数，格式为 `generate_training_data_nonpv 参数1名 参数1值 参数2名 参数2值 ...`。

目前可用的选项如下：

`depth` - 用于重新评分的搜索深度。默认值：3。

`count` - 要生成的训练数据条目数。1 个条目 == 1 个局面。默认值：1,000,000 (100万)。

`exploration_min_nodes` - 自对弈探索期间使用的最小节点数。默认值：5000。

`exploration_max_nodes` - 自对弈探索期间使用的最大节点数。节点数将在最小值和最大值之间均匀分布选择。默认值：15000。

`exploration_save_rate` - 在探索性自对弈期间遇到的局面中，被保存用于后续重新评分的比例。默认值：0.01 (意味着搜索中每 100 个局面就有 1 个被保存用于重新评分)。

`output_file` - 输出文件名。如果扩展名不存在或与所选的训练数据格式不匹配，将自动附加正确的扩展名。默认值：`generated_gensfen_nonpv`。

`eval_limit` - 绝对值超过此评估值的局面将不会被写入，并且会终止当前的自对弈对局。不应超过 10000 (VALUE_KNOWN_WIN)，但硬性上限为 2 步杀 (约 30000)。默认值：4000。

`exploration_eval_limit` - 与 `eval_limit` 相同，但在探索期间使用固定深度搜索的值。

`exploration_min_pieces` - 在自对弈中开始固定深度搜索的最小棋子数。请注意，即使棋盘上有 N 个棋子，固定节点搜索通常也会达到棋子更少的局面，这些局面也会被保存。默认值：8。

`exploration_max_ply` - 探索性自对弈的最大回合数。默认值：200。

`smart_fen_skipping` - 这是一个标志选项。指定后，一些不适合作为教学材料的局面将从输出中移除。这包括最佳着法是吃子或升变，以及王被将军的局面。

`book` - 用于起始局面的开局库路径。目前仅支持 .epd 格式。如果未指定，则始终使用标准国际象棋起始局面。

`data_format` - 使用的训练数据格式。可以是 `bin` 或 `binpack`。默认值：`binpack`。

`seed` - 伪随机数生成器 (PRNG) 的种子。可以是一个数字或一个字符串。如果是字符串，则使用其哈希值。如果未指定，则使用当前时间。
