This section explains what the `easy_train.py` script actually does, its quirks, and how to use if effectively. The script can be found in the script directory of this repo, and is the only necessary file to proceed.

# What is `easy_train.py`

`easy_train.py` is a python script that manages combined network training and network testing in an organized and self-contained way. It is made to require only minimal knowledge to run according to existing specification while also keeping customizability for experienced users. The output is aggregated and displayed either in a TUI (Terminal User Interface) for human user consumption or directly into the terminal to allow pipes. It is tested on Windows (>=7) and Linux.

![easy_train.py TUI](https://user-images.githubusercontent.com/8037982/176677909-7d36b6e1-6ac9-43f1-8965-c6394988cdca.png)

## Basic concepts

The script organizes training sessions into "experiments". Each "experiment" sets up the trainer, players, logging, and other prerequisites (for example start network models) locally, so that it's possible to determine exactly how an experiment was ran. Experiments can be resumed mid-way so they are not strictly individual sessions.

Even though the `easy_train.py` script is placed in the `scripts` directory of the `nnue-pytorch` repo, it does not require anything else to from the repo to run. It a single, self-contained python source file.

## Dependencies and requirements

On Windows MSYS2 with GCC installed is required. It can be installed easily following https://packages.msys2.org/groups/mingw-w64-x86_64-toolchain. The compiler must be callable from the command line by just `gcc`, i.e. the folder with its binaries must be added to the PATH environmental variable.

TODO: what globally available software is required. Clarify msys2 requirement/setup

## Setup instructions

### Windows

TODO: detailed step by step how to set this up from a clean system install

### Linux

TODO: detailed step by step how to set this up from a clean system install

## Workspace directory structure

The script creates a workspace in the specified path on first invocation. This workspace has the following directory tree:

- **workspace_dir**
  - **books** - contains all the .epd opening books that were used at least once. May also contain then in an archived format, depending on how they were acquired.
  - **c-chess-cli** - contains a clone of [c-chess-cli](https://github.com/lucasart/c-chess-cli) at a specific commit to minimize compatibility issues. It is built from source on first use.
  - **ordo** - contains a clone of [ordo](https://github.com/michiguel/Ordo) at a specific commit to minimize compatibility issues. It is built from source on first use.
  - **experiments** - contains all training sessions (experiments), one directory per session
    - **experiment_1** - contains all the data for a single experiment, here the experiment is named "1"
      - **logging** - contains a dump of the CLI arguments and an execution log
      - **nnue-pytorch** - contains a clone of a specified trainer repository. All the tasks that this script manages use scripts from this directory. The data loader is built on first use.
      - **start_models** - if the training is based on some other experiment, or an existing model, this directory will contain the model that is used to seed the training
      - **stockfish_base** - contains a clone of a specified repo containing the baseline playing engine. It is expected to be stockfish-like. The player is built from source on first use.
      - **stockfish_test** - contains a clone of a specified repo containing the playing engine that corresponds to the networks being produced. In particular it will be a different player than the baseline when the network architecture changes. It is expected to be stockfish-like. The player is built from source on first use.
      - **training** - contains the trainer output, including tensorboard events, network checkpoints, c-chess-cli output, ordo output
        - **run_0** - contains output from a single train.py instance, in particular converted networks in .nnue format
          - **lightning_logs**
            - **version_0** - usually only version "0" will be present. One directory is created for each time training is invoked with the same log directory, this a pytorch-lightning thing. This directory contains the tfevents file that's readable by tensorboard.
              - **checkpoints** - contains raw .ckpt full checkpoints of the training, including optimizer state. These can be converted to .pt/.nnue or used directory for further training.
        - **run_1** - each run gets a separate directory, and it is possible to do multiple runs per experiment. If that's the case then multiple "run_\*" directories are present.
          - **...**
        - **...**
    - **experiment_2**
      - **...**
    - **...**

## Basic behaviour switches

TODO: from scratch/continue/retrain

## Example invocation

The example invocations (.bat and .sh) present in the script directory are NOT production ready. They are there merely to show most important available options and run for testing purposes. Below is an invocation that should be close to master, with comments on important options (.sh).

First training session - training a network from scratch:

```
python easy_train.py \
    --training-dataset=nodes5000pv2_UHO.binpack \ # see wiki on datasets
    --validation-dataset=nodes5000pv2_UHO.binpack \ # see wiki on datasets
    --num-workers=4 \ # enough to get good speed on pretty much all gpus
    --threads=2 \ # enough to get good speed on pretty much all gpus
    --gpus="0," \ # use only the first gpu, no multi-gpu training as of right now; if multiple gpus are specified here then just more runs will be done in parallel
    --runs-per-gpu=1 \ # you can increase it if your gpu is not saturated
    --batch-size=16384 \
    --max_epoch=600 \ # nets start getting saturated at around epoch 400
    --do-network-training=True \
    --do-network-testing=True \
    --tui=True \
    --network-save-period=10 \ # save every 10th net, you might change it depending on how much storage you're willing to use
    --random-fen-skipping=3 \
    --start-lambda=1.0 \
    --end-lambda=0.75 \
    --gamma=0.992 \ # default gamma, determines how learning rate drops after each epoch
    --lr="8.75e-4" \ # default learning rate
    --fail-on-experiment-exists=True \
    --build-engine-arch=x86-64-modern \ # you might change this to an other architecture if you CPU supports it (this is the ARCH parameter of the stockfish makefile)
    --build-threads=2 \
    --epoch-size=100000000 \ # a pretty standard value used for a long time, no need to change this, very max_epoch instead
    --validation-size=1000000 \ # validation step is not necessary so we put little work here
    --network-testing-threads=24 \ # as many as you can afford, will slow down training due to strain on memory
    --network-testing-explore-factor=1.5 \
    --network-testing-book="https://github.com/official-stockfish/books/raw/master/UHO_XXL_+0.90_+1.19.epd.zip" \
    --network-testing-nodes-per-move=20000 \ # important to use nodes instead of time for more consistent results
    --network-testing-hash-mb=8 \
    --network-testing-games-per-round=200 \
    --engine-base-branch=official-stockfish/Stockfish/master \ # setup the branch relevant to your training run
    --engine-test-branch=official-stockfish/Stockfish/master \ # setup the branch relevant to your training run
    --nnue-pytorch-branch=glinscott/nnue-pytorch/master \ # setup the branch relevant to your training run
    --workspace-path=./easy_train_data \ # change to wherever you want all data to be stored
    --experiment-name=test # change to whatever you like, don't use whitespaces
```

Second training session - fine tuning an already existing network with better data:

TODO: needs verification

```
python easy_train.py \
    --training-dataset=Leela-dfrc_n5000.binpack \ # see wiki on datasets
    --validation-dataset=Leela-dfrc_n5000.binpack \ # see wiki on datasets
    --num-workers=4 \ # enough to get good speed on pretty much all gpus
    --threads=2 \ # enough to get good speed on pretty much all gpus
    --gpus="0," \ # use only the first gpu, no multi-gpu training as of right now; if multiple gpus are specified here then just more runs will be done in parallel
    --runs-per-gpu=1 \ # you can increase it if your gpu is not saturated
    --start-from-experiment=test \ # the name of the experiment to retrain from, will choose the best/latest network. Alternatively use --start-from-model
    --batch-size=16384 \
    --max_epoch=800 \ # nets start getting saturated at around epoch 400, but give it some more time now since we're using lower LR and a slower schedule
    --do-network-training=True \
    --do-network-testing=True \
    --tui=True \
    --network-save-period=10 \ # save every 10th net, you might change it depending on how much storage you're willing to use
    --random-fen-skipping=3 \
    --start-lambda=1.0 \
    --end-lambda=0.75 \
    --gamma=0.995 \ # reduce LR slower than in the first training session
    --lr="4.375e-4" \ # half the start LR
    --fail-on-experiment-exists=True \
    --build-engine-arch=x86-64-modern \ # you might change this to an other architecture if you CPU supports it (this is the ARCH parameter of the stockfish makefile)
    --build-threads=2 \
    --epoch-size=100000000 \ # a pretty standard value used for a long time, no need to change this, very max_epoch instead
    --validation-size=1000000 \ # validation step is not necessary so we put little work here
    --network-testing-threads=24 \ # as many as you can afford, will slow down training due to strain on memory
    --network-testing-explore-factor=1.5 \
    --network-testing-book="https://github.com/official-stockfish/books/raw/master/UHO_XXL_+0.90_+1.19.epd.zip" \
    --network-testing-nodes-per-move=20000 \ # important to use nodes instead of time for more consistent results
    --network-testing-hash-mb=8 \
    --network-testing-games-per-round=200 \
    --engine-base-branch=official-stockfish/Stockfish/master \ # setup the branch relevant to your training run
    --engine-test-branch=official-stockfish/Stockfish/master \ # setup the branch relevant to your training run
    --nnue-pytorch-branch=glinscott/nnue-pytorch/master \ # setup the branch relevant to your training run
    --workspace-path=./easy_train_data \ # change to wherever you want all data to be stored
    --experiment-name=test_retrain # change to whatever you like, don't use whitespaces
```