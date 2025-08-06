# Binpack

Binpack 是一种二进制训练数据存储格式，其设计旨在利用仅相差一步棋的局面链。因此，它非常适合紧凑地存储从真实对局中生成的数据（与例如来自开局库的随机局面相反）。

它目前通过 `extra/nnue_data_binpack_format.h` 中的单个头文件库实现。

下面是该格式的粗略描述，使用类似 BNF 的表示法。

```cpp
[[nodiscard]] std::uint16_t signedToUnsigned(std::int16_t a) {
    std::uint16_t r;
    std::memcpy(&r, &a, sizeof(std::uint16_t));
    if (r & 0x8000) r ^= 0x7FFF; // 如果为负，则翻转值位
    r = (r << 1) | (r >> 15); // 将符号位存储在第 0 位
    return r;
}

file := <block>*
block := BINP<chain>*
chain := <stem><movetext>
stem := <pos><move><score><ply_and_result><rule50> (32 字节)
pos := https://github.com/Sopel97/nnue_data_compress/blob/master/src/chess/Position.h#L1166 (24 字节)
move := https://github.com/Sopel97/nnue_data_compress/blob/master/src/chess/Chess.h#L1044 (2 字节)
score := signedToUnsigned(score) (2 字节, 大端序)
ply_and_result := ply bitwise_or (signedToUnsigned(result) << 14) (2 字节, 大端序)
rule50 := rule_50_counter (2 字节, 大端序)
    // 这是旧版本的一个小缺陷，
    我不想破坏向后兼容性。实际上，这意味着未来
    还剩下一个字节可用于其他用途，因为 rule50 总是能装入一个字节。

movetext := <count><move_and_score>*
count := movetext 中的半回合数 (2 字节, 大端序)。可以为 0。
move_and_score := <encoded_move><encoded_score> (约 2 字节)
encoded_move := 噢，这个解释起来有点复杂。
    https://github.com/Sopel97/nnue_data_compress/blob/master/src/compress_file.cpp#L827
    https://github.com/Sopel97/chess_pos_db/blob/master/docs/bcgn/variable_length.md

encoded_score := https://en.wikipedia.org/wiki/Variable-width_encoding
    块大小为 4 位 + 1 位扩展位。
    编码值为 signedToUnsigned(-prev_score - current_score)
    (分数总是从 <pos> 中轮到走棋的一方的角度来看，这就是为什么 prev_score 前面有‘-’)
```
