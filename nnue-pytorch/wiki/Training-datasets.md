# What makes a good dataset

We don't really know. Datasets with more "positional" evaluations might be better, as we find datasets derived from Lc0 training data to be good. Better evaluations don't always give better results, as it's a tradeoff between quality and learnability. Good datasets are chosen empirically.

# Generating datasets

## Stockfish data generator

Stockfish has a [branch](https://github.com/official-stockfish/Stockfish/tree/tools) with various tools, in particular a training data generator. The generator works by doing self-play games and collecting positions and their evaluations along the way. For details on usage you can see the docs in the linked branch. The last good [dataset generated in this way](https://drive.google.com/file/d/1lFC_tej8WyXojhh7-AmXV_kkrTauEsqt/view?usp=sharing) consists of 16B positions and was done by vondele with the following script:
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

The noob_3moves.epd opening book can be found [in the Stockfish book repo](https://github.com/official-stockfish/books).

## Lc0 data converter

The [Lc0 data rescorer](https://github.com/Tilps/lc0/tree/rescore_tb) has a functionality to convert the data to .plain format that's used for NNUE datasets. Datasets produced in this way are of higher quality than the ones generated with Stockfish, and generally end up producing better networks.

### Installing the Lc0 data converter

TODO: this

### Running the Lc0 data converter

Borg, who created the functionality, originally used this script to create the first batch of data derived from Lc0
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

Right now Chess960 is not supported, and the source data must not contain Chess960 positions.

# Good datasets

* [large_gensfen_multipvdiff_100_d9.binpack](https://drive.google.com/file/d/1VlhnHL8f-20AXhGkILujnNXHwy9T-MQw/view?usp=sharing) - the original depth 9 dataset generated with Stockfish. It's still ok, but superseded by the more recent one listed below.
* [data_d9_2021_09_02.binpack](https://drive.google.com/file/d/1lFC_tej8WyXojhh7-AmXV_kkrTauEsqt/view?usp=sharing) - the most recent dataset generated with Stockfish; depth 9 still seems to be the sweet spot.
* [nodes5000pv2_UHO.binpack](https://drive.google.com/file/d/1UQdZN_LWQ265spwTBwDKo0t1WjSJKvWY/view?usp=sharing) - a recent attempt to use fixed nodes instead of fixed depth for data generation using Stockfish. It produces the best playing nets for [the UHO books.](https://www.sp-cc.de/uho_xxl_project.htm) and at least on par with the depth 9 dataset for others.
* [training_data.binpack](https://drive.google.com/file/d/1RFkQES3DpsiJqsOtUshENtzPfFgUmEff/view) - the first good dataset derived from Lc0. It was used to beat the then master Stockfish net.
* [T60T70wIsRightFarseer.binpack](https://drive.google.com/file/d/1_sQoWBl31WAxNXma2v45004CIVltytP8/view) - a mix of various datasets, including Lc0 T60, T70 data, Stockfish self-play data from openings it usually gets wrong, and some more converted Lc0 data from Farseer. It is currently one of the best datasets available.
* [dfrc_n5000.binpack](https://drive.google.com/file/d/17vDaff9LAsVo_1OfsgWAIYqJtqR8aHlm/view) - new data generated with stockfish at 5000 nodes per move, from [DFRC](https://www.schemingmind.com/home/knowledgebase.aspx?article_id=123) [opening book](https://github.com/official-stockfish/books/blob/master/DFRC_openings.epd.zip). Not used in isolation.
* Leela-dfrc_n5000.binpack - made by running `python3` [interleave_binpacks.py](https://github.com/official-stockfish/Stockfish/blob/tools/script/interleave_binpacks.py) `T60T70wIsRightFarseer.binpack dfrc_n5000.binpack Leela-dfrc_n5000.binpack`. Used for retraining the network.

TODO: https://robotmoon.com/nnue-training-data/, inquire linrock about exact datasets used and the process, replicate from zero if needed

# Using datasets in practice

Usually it is not enough to train a network on one dataset only. Moreover, the order of the datasets trained with matters. It is usually best to first train a network with datasets generated with Stockfish (depth 9, nodes 5000), and then retrain an already good network using various Lc0-derived datasets. Training solely on the Lc0-derived datasets doesn't produce as good results, presumably either to position coverage or the concepts are too hard to learn from scratch. How to retrain a network with a different dataset can be found in the "Basic Training Procedure" section of this wiki.

# Historical datasets used for Stockfish master network progression.
TODO: this