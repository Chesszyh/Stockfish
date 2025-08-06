# 什么造就了一个好的数据集

我们其实并不清楚。拥有更多“局面性”评估的数据集可能会更好，因为我们发现源自 Lc0 训练数据的数据集就很好。更好的评估并不总是能带来更好的结果，因为这是质量和可学习性之间的权衡。好的数据集是根据经验选择的。

# 生成数据集

## Stockfish 数据生成器

Stockfish 有一个包含各种工具的[分支](https://github.com/official-stockfish/Stockfish/tree/tools)，特别是训练数据生成器。该生成器通过进行自我对弈并沿途收集局面及其评估来工作。有关用法详情，您可以查看链接分支中的文档。最后一个以这种方式生成的优秀[数据集](https://drive.google.com/file/d/1lFC_tej8WyXojhh7-AmXV_kkrTauEsqt/view?usp=sharing)包含 160 亿个局面，由 vondele 使用以下脚本完成：
```
dir=data
fulldir=${dir}_${diff}_d${DEPTH}

mkdir -p ${fulldir}

options="
uci
setoption name PruneAtShallowDepth value false
setoption name Use NNUE value true
setoption name Threads value 250
setoption name Hash value 10240
isready
generate_training_data depth 9 count 18000000000 random_multi_pv 4 random_multi_pv_diff 100 set_recommended_uci_options data_format binpack output_file_name ${fulldir}/gensfen.binpack book noob_3moves.epd seed ${RANDOM}${RANDOM}
quit"

echo "$options"
 
printf "$options" | ./stockfish > ${fulldir}/out.txt

echo "Done ${TID}"
```

noob_3moves.epd 开局库可以在 [Stockfish 开局库仓库](https://github.com/official-stockfish/books)中找到。

## Lc0 数据转换器

[Lc0 数据重计分器](https://github.com/Tilps/lc0/tree/rescore_tb)具有将数据转换为用于 NNUE 数据集 .plain 格式的功能。以这种方式生成的数据集比使用 Stockfish 生成的数据集质量更高，并且通常最终会产生更好的网络。

### 安装 Lc0 数据转换器

待办事项：此部分

### 运行 Lc0 数据转换器

创建该功能的 Borg 最初使用此脚本创建了第一批源自 Lc0 的数据
```
#!/usr/bin/env bash
set -e

function func () {
  ./rescorer rescore --input=$1 --syzygy-paths=../syzygy --nnue-plain-file=$1.plain --threads=4
  rm -rf $1
  ./stockfish convert $1.plain trainingdata/$1.binpack validate
  rm $1.plain
}

for FILE in training-run1-test60-202106{01..10}-{00..23}17
#for FILE in training-run2-test74-20210505-{00..23}{16..23}
do
  wget https://storage.lczero.org/files/training_data/test60/$FILE.tar || true
#  wget https://storage.lczero.org/files/training_data/test74/$FILE.tar || true
  if [ -f $FILE.tar ]
  then
    tar xf $FILE.tar
    rm $FILE.tar
    func $FILE &
  fi
done
```

目前不支持 Chess960，源数据不得包含 Chess960 局面。

# 好的数据集

* [large_gensfen_multipvdiff_100_d9.binpack](https://drive.google.com/file/d/1VlhnHL8f-20AXhGkILujnNXHwy9T-MQw/view?usp=sharing) - 使用 Stockfish 生成的原始深度 9 数据集。它仍然可以，但已被下面列出的更新版本所取代。
* [data_d9_2021_09_02.binpack](https://drive.google.com/file/d/1lFC_tej8WyXojhh7-AmXV_kkrTauEsqt/view?usp=sharing) - 使用 Stockfish 生成的最新数据集；深度 9 似乎仍然是最佳选择。
* [nodes5000pv2_UHO.binpack](https://drive.google.com/file/d/1UQdZN_LWQ265spwTBwDKo0t1WjSJKvWY/view?usp=sharing) - 最近一次尝试使用固定节点而不是固定深度来使用 Stockfish 生成数据。它为 [UHO 开局库](https://www.sp-cc.de/uho_xxl_project.htm)生成了最佳的对弈网络，并且至少与其他深度 9 数据集相当。
* [training_data.binpack](https://drive.google.com/file/d/1RFkQES3DpsiJqsOtUshENtzPfFgUmEff/view) - 第一个源自 Lc0 的优秀数据集。它被用来击败当时的大师级 Stockfish 网络。
* [T60T70wIsRightFarseer.binpack](https://drive.google.com/file/d/1_sQoWBl31WAxNXma2v45004CIVltytP8/view) - 各种数据集的混合，包括 Lc0 T60、T70 数据、来自 Stockfish 通常会出错的开局的自我对弈数据，以及一些来自 Farseer 的转换后的 Lc0 数据。它目前是可用的最佳数据集之一。
* [dfrc_n5000.binpack](https://drive.google.com/file/d/17vDaff9LAsVo_1OfsgWAIYqJtqR8aHlm/view) - 使用 stockfish 以每步 5000 个节点从 [DFRC](https://www.schemingmind.com/home/knowledgebase.aspx?article_id=123) [开局库](https://github.com/official-stockfish/books/blob/master/DFRC_openings.epd.zip)生成的新数据。不单独使用。
* Leela-dfrc_n5000.binpack - 通过运行 `python3` [interleave_binpacks.py](https://github.com/official-stockfish/Stockfish/blob/tools/script/interleave_binpacks.py) `T60T70wIsRightFarseer.binpack dfrc_n5000.binpack Leela-dfrc_n5000.binpack` 创建。用于重新训练网络。

待办事项：https://robotmoon.com/nnue-training-data/，向 linrock 询问所使用的确切数据集和过程，如果需要，从零开始复制

# 在实践中使用数据集

通常，仅在一个数据集上训练网络是不够的。此外，训练所用数据集的顺序也很重要。通常最好先使用 Stockfish 生成的数据集（深度 9，节点 5000）训练网络，然后使用各种源自 Lc0 的数据集重新训练已经很好的网络。仅在源自 Lc0 的数据集上训练并不能产生同样好的结果，大概是因为局面覆盖率的原因，或者概念太难从头开始学习。如何使用不同的数据集重新训练网络可以在本 wiki 的“基本训练过程”部分找到。

# 用于 Stockfish 主网络进展的历史数据集。
待办事项：此部分