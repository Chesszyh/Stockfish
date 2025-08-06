It is important to understand what tools are available and how they work. In this section individual components of the trainer and and other utilities will be described.

# List of components
## Main components

* train.py - the entry point for training the networks
* model.py - contains the definition of the network's architecture and describes the inference process
* serialize.py - handles serializing/deserializing the model defined in model.py and conversion between .nnue, .pt, .ckpt serialized models.
* features.py, halfka.py, halfka_v2.py, halfka_v2_hm.py, halfkp.py - descriptions of network's inputs
* training_data_loader.cpp - handles loading the training data (.bin, .binpack) and prepares batches
* nnue_dataset.py - a broker between the native data loader and the trainer
* feature_transformer.py - a highly optimized CUDA implementation of the first layer (fully connected with sparse inputs)

## Utility and tools

* cross_check_eval.py - a utility script for checking correspondence between pytorch model evaluation and stockfish player evaluation using a .nnue model
* visualize.py - provides comprehensive visualizations for a single network [against a baseline]
* visualize_multi_hist.py - provides comparative visualizations for multiple networks
* run_games.py - provides a way to run games using c-chess-cli for networks produced as the training progresses
* do_plots.py - creates plots of the event data (training loss, validation loss, ordo elo) gathered during training
* delete_bad_nets.py - deletes worst networks based on ordo elo

# train.py

This component is an entry point for the training procedure. It identifies the device to train with, assembles training parameters, creates the initial (or loads an already existing) model, creates data loaders (and defines the epoch size), sets up the tensorboard logger, and starts the training process.

To see what invocation parameters are available one can run `python train.py --help`.

The notion of an epoch in this trainer is slightly different from the usual definition, here we define an epoch as 100 million samples. This is because the training datasets can vary wildly in size.

The training process is indefinite unless stated otherwise (`--max_epochs`). Checkpoints are saved in .ckpt format in the specified log directory (see invocation parameters). When and which checkpoints are saved can be modified through the code by changing the parameters passed to the `pl.callbacks.ModelCheckpoint`. Individual checkpoints in .ckpt format store both the model and the optimizer state, therefore they are quite sizable, it is advised to save only every few epochs.

# model.py

The model is defined as a `pl.LightningModule` named `NNUE`. To better understand the low-level structure of the model it is suggested to read [this documentation](https://github.com/glinscott/nnue-pytorch/blob/master/docs/nnue.md). 

The model defines the inference process and chooses the parameters to be optimized. To account for quantization it clips weights of some layers after each step to the supported range. The training loop is handled by pytorch-lightning, which interfaces with the model through certain methods of the model.

The model also contains code that allows it to convert between some feature sets when loading existing models. Notably it is possible to add virtual features after the existing model is loaded. The model module also exposes a function to coalesce such virtual feature weights into real feature weights.

# serialize.py

While this component is used by the trainer to load existing models it can also be used as a standalone script for conversion between model formats. It is necessary to convert the network to .nnue format before the playing engine can use it. It supports conversion between the following formats:
* .nnue - the format used by the playing engines. It contains the net in a quantized form and doesn't preserve the optimizer state.
* .pt - the pytorch's format. It doesn't preserve the optimizer state so it is useful when trying to restart the training with changes to the optimizer. It stores the net using full precision.
* .ckpt - the format that the checkpoints are stored in. Only networks of .ckpt format are produced by the trainer. Networks saved like this contain full optimizer state and the state of the trainer, for example the current epoch. This format can be useful for temporarily suspending the training.

The serializer is heavily coupled with the model. Changes to the model usually require corresponding changes to the serializer. Also, only the serializer knows how to perform quantization when converting to the .nnue format, and it must correspond to the quantized implementation in the playing engine.

# features.py, halfka.py, halfka_v2.py, halfka_v2_hm.py, halfkp.py

The features.py file imports all available feature sets. Individual feature sets define their parts (feature blocks) and their sizes, provide initial PSQT values for each feature, and allow retrieving the factors of each feature (all features (real or virtual) which correspond to a given real feature). On the python side it is not necessary for the decomposition of the board state to the list of features to be implemented.

# training_data_loader.cpp

Due to performance requirements it is necessary to have the training data loader implemented natively. The data loader can be compiled with `compile_data_loader.bat` which uses cmake. One can also compile without cmake if they know how to.

This component implements the decomposition of board states to individual features. This step must be coherent with the feature sets defined on the python side.

The data loader provides whole batches at a time, to remove the cost of concatenating individual samples into a batch later on. `SparseBatch` represents such a batch. It allocates one array for each property of the sample. Features are stored as (index, value) pairs for active features. The upper bound on the number of active features in a position is known. Unused features are stored with index `-1`.

The data loader supports multiple threads. The amount of threads can be specified on creation and is split [heristically] between threads reading the data from disk and threads forming batches.

The data loader also supports forming an array of fen strings instead of training batches. This is useful for other utilities such as `cross_check_eval.py`.

The bindings are done using ctypes in `nnue_dataset.py` component.

# nnue_dataset.py

Contains the definitions of the data loaders compatible with pytorch and uses ctypes bindings to make them use the native data loader. The tensors from the batch are copied to the main training device and reformed into pytorch tensors.

It requires the shared library of the compiled native data loader to be present in the scripts directory as `training_data_loader.[so|dll|dylib]`.

# feature_transformer.py

This file contains a CUDA implementation (using CuPY) of the first layer of the network. We take advantage of the very sparse input in a very specific way, which speeds up the training considerably compared to using pytorch's sparse tensors. More information about this can be found [here](https://github.com/glinscott/nnue-pytorch/blob/master/docs/nnue.md#optimizing-the-trainer-cuda)