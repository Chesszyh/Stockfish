# generate_training_data (生成训练数据)

`generate_training_data` 命令允许通过自对弈生成训练数据，其方式比传统对局更适合训练。它引入随机着法以丰富开局多样性，并采用固定深度进行局面评估。

与 Stockfish 中的所有命令一样，`generate_training_data` 既可以从命令行调用（例如 `stockfish.exe generate_training_data ...`，但不推荐这样做，因为在命令执行前无法设置 UCI 选项），也可以在交互式提示符中调用。

建议将 `PruneAtShallowDepth` UCI 选项设置为 `false`，这将提高固定深度搜索的质量。

建议将 `EnableTranspositionTable` UCI 选项保持为默认值 `true`，这会加快生成过程，且不会显著损害数据的均匀性。

`generate_training_data` 命令接受命名参数，格式为 `generate_training_data 参数1名 参数1值 参数2名 参数2值 ...`。

目前可用的选项如下：

`set_recommended_uci_options` - 这是一个修饰符而非参数，后面不跟值。如果指定，一些 UCI 选项将被设置为推荐值。

`depth` - 设置评估每个局面的最小和最大深度。默认值：3。

`mindepth` - 评估每个局面的最小深度。如果未指定，则与 `depth` 相同。

`maxdepth` - 评估每个局面的最大深度。如果未指定，则与 `depth` 相同。

`nodes` - 用于评估每个局面的节点数。该数值会乘以当前搜索的 PV (主变) 数量。此选项不会覆盖 `depth` 和 `depth2` 选项。如果指定，则深度或节点数限制以先达到者为准。

`count` - 要生成的训练数据条目数。1 个条目 == 1 个局面。如果同时指定了 `count` 和 `max_time_*`，则当任一条件满足时，数据生成过程将结束。默认值：8,000,000,000 (80亿)。

`max_time_seconds`, `max_time_minutes`, `max_time_hours` - 指定数据生成的最长运行时间。正在进行的自对弈对局不会被中断。如果同时指定了 `count` 和 `max_time_*`，则当任一条件满足时，数据生成过程将结束。默认值：约 250 年。

`output_file_name` - 输出文件名。如果扩展名不存在或与所选的训练数据格式不匹配，将自动附加正确的扩展名。默认值：`generated_kifu`。

`eval_limit` - 绝对值超过此评估值的局面将不会被写入，并且会终止当前的自对弈对局。不应超过 10000 (VALUE_KNOWN_WIN)，但硬性上限为 2 步杀 (约 30000)。默认值：3000。

`random_move_min_ply` - 可以执行随机着法（而非搜索选择的着法）的最小回合数。默认值：1。

`random_move_max_ply` - 可以执行随机着法的最大回合数。默认值：24。

`random_move_count` - 单局自对弈中随机着法的最大数量。默认值：5。

`random_move_like_apery` - 值为 0 或 1。如果为 1，则随机的王移动有 50% 的概率会尽可能地被对手的随机王移动跟随。默认值：0。

`random_multi_pv` - 用于确定随机着法的 PV (主变) 数量。如果未指定，则选择一个真正随机的着法。如果指定，将执行多 PV 搜索，随机着法将是搜索选择的着法之一。

`random_multi_pv_diff` - 使得多 PV 随机着法选择只考虑那些比次优着法差值不超过 `random_multi_pv_diff` 的着法。默认值：30000 (所有多 PV 着法)。

`random_multi_pv_depth` - 用于随机着法多 PV 搜索的深度。默认值：`depth2`。

`random_multi_pv_nodes` - 用于随机着法多 PV 搜索的最大节点数。默认值：`nodes`。

`write_min_ply` - 将被输出的训练数据条目的最小回合数。默认值：16。

`write_max_ply` - 将被输出的训练数据条目的最大回合数。默认值：400。

`book` - 用于起始局面的开局库路径。目前仅支持 .epd 格式。如果未指定，则始终使用标准国际象棋起始局面。

`save_every` - 每个文件保存的训练数据条目数。如果未指定，则始终只生成一个文件。如果指定，可能会生成多个文件（每个文件最多包含 `save_every` 个条目），并且每个文件都会附加一个唯一的编号。

`random_file_name` - 如果指定，输出文件名将被随机选择。此选项会覆盖 `output_file_name`。

`keep_draws` - 值为 0 或 1。如果为 1，则和棋对局的训练数据也将被输出。默认值：1。

`adjudicate_draws_by_score` - 值为 0 或 1。如果为 1，则在第 80 回合后，当局面评估值连续至少 8 个半回合保持为 0 时，对局将被裁定为和棋。默认值：1。

`adjudicate_draws_by_insufficient_mating_material` - 值为 0 或 1。如果为 1，则因物质不足无法将杀的局面将被裁定为和棋。默认值：1。

`data_format` - 使用的训练数据格式。可以是 `bin` 或 `binpack`。默认值：`binpack`。

`seed` - 伪随机数生成器 (PRNG) 的种子。可以是一个数字或一个字符串。如果是字符串，则使用其哈希值。如果未指定，则使用当前时间。
