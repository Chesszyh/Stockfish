This section assumes that the libraries are installed and the trainer can be ran. Test by running `python train.py --help`. It also assumes that the user already has the training data they want to use for training.

# Running the trainer

The way the trainer should be invoked depends on the setup and data. Below a sample invocation is presented, and later each parameter will be expanded upon.

```
python3 train.py \
    ./training/data/large_gensfen_multipvdiff_100_d9.binpack \
    ./training/data/large_gensfen_multipvdiff_100_d9.binpack \
    --gpus "0," \
    --threads 4 \
    --num-workers 4 \
    --batch-size 16384 \
    --progress_bar_refresh_rate 20 \
    --random-fen-skipping 3 \
    --features=HalfKAv2_hm^ \
    --lambda=1.0 \
    --max_epochs=400 \
    --default_root_dir ./training/runs/run_0
```

The first two parameters are the training dataset and validation dataset respectively. The datasets are read in a cyclic manner (indefinitely). The training dataset is used to calculate the gradients and perform backpropagation step to train the network. The validation dataset is used between the training stages to check the performance of the network. These datasets may or may not be different. The training (and validation) results will depend heavily on the chosen datasets.

The `--gpus` parameter controls the device[s] to use for training. Currently only single-GPU training is supported. The way this parameter is specified can be found [here](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#gpus). In the above example the trainer is instructed to use only the GPU with id 0.

The `--threads` parameter controls the number of threads pytorch will use. The best value for this parameter depends on hardware, generally 4 is more than enough. It is advised to benchmark with values between 1 and 8 on each hardware.

The `--num-workers` parameter controls the number of threads the data loader will use. The threads are split between reading the data from disk and forming batches. At least 2 workers are only used (one for reading from disk and one for forming batches), so the values below 3 have no effect. It is advised to benchmark with values between 1 and 8 on each hardware.

Example benchmark results on a A100:
| iterations/s <br> threads >> <br> workers VV |     1 |     2 |     4 |     8 |    16 |
|--------------------------------------------------------|------:|------:|------:|------:|------:|
|                                                      1 | 66.97 | 67.46 | 67.94 | 65.55 | 65.55 |
|                                                      4 | 80.40 | 83.64 | 85.38 | 86.91 | 87.57 |
|                                                      8 | 76.58 | 82.52 | 86.00 | 86.69 | 85.15 |
|                                                     16 | 78.12 |  82.5 | 84.65 | 84.98 | 86.00 |

The `--batch_size` parameter controls the size of the sample batch. Larger batches require more GPU memory but may be faster on high-end GPUs. The batch size and the learning rate depend on each other. It is advised to use batch size of 16384.

The `--progress_bar_refresh_rate` parameter controls the refresh rate of the console output produced by the trainer. The value passed is how many iterations (steps) must pass before the next refresh. Lower values may slow down the training a little or give visible flickering when using high-end GPUs.

The `--random-fen-skipping` parameter controls the filtering rate of the training samples. The higher the value the higher the probability that a sample is skipped while reading. Higher values generally make the trainer see more diverse data in shorter timespan, but it may slow down the training due to increased load on the data loader.

The `--features` parameter controls the feature set to be used as the network's inputs. The default value usually corresponds to the Stockfish's master net.

The `--lambda` parameter controls whether the trainer uses the outcome (game result) or the score (eval) as a training target. `lambda` of 1.0 means the use purely the score. `lambda` of 0.0 means to use purely the outcome. Values in between provide a proportional mix of the two.

The `--max_epochs` parameter specifies after how many epochs the training terminates. If this parameter is not specified then the training must be stopped manually.

The `--default_root_dir` specifies the path where the checkpoints and tensorflow events are to be saved.

## Additional parameters

The `--no-smart-fen-skipping` parameter disables the "intelligent" sample filtering done by the data loader. It is generally known that for some reason positions with tactical captures are not very well understood by the network and training on them results in worse networks. This is alleviated partially by the "smart fen skipping", which removes the samples where the bestmove is a capture. Note that this is not an ideal solution, but no better solution has been presented yet.

The `--no-wld-fen-skipping` parameter disables stochastic filtering of samples based on their eval correlation to outcomes. It was found that for some datasets (usually involving lc0 training data) it is beneficial to sometimes skip samples where the evaluation doesn't correlate with the end game result. For some datasets this filtering might not be a positive so there is a way to disable it. For some datasets it might slow the training down or require specifying more workers. Disabling it is not a big change but it is measurable for some datasets.

## Restarting from an existing model

### Restarting from a .nnue/.pt

To restart from a .nnue first it needs to be converted to .pt. This can be done by invoking `python serialize.py --features=<feature set of the .nnue file> nn.nnue nn.pt`.

The `--resume-from-model` parameter specifies a path to the .pt model to load as the initial training model. Since it doesn't save the training state nor optimizer state the training is restarted from scratch, with just the weights and biases loaded from the .pt model instead of being reinitialized.

The `--features` parameter used during training must match the feature set of the .pt model; it may add optionally add virtual features (for example one can train with --features=HalfKA^ when the .pt model has only HalfKA features. This is common when restarting from .nnue models which lose the virtual feature information).

This is useful for retraining existing nets using different datasets or if changing the optimizer settings is required. 

It is known that for some reason high quality datasets (like for example lc0) achieve better results when used to retrain a network trained on lower quality datasets (like for example generated with Stockfish at low depth) than when training on the high quality datasets from zero.

### Restarting from a .ckpt

.ckpt format contains the optimizer and the trainer state, therefore it is ideal for resuming training after it was suspended for some reason.

The `--resume_from_checkpoint` parameter specifies a path to the .ckpt checkpoint file. The training resumes from the epoch the checkpoint was taken at. 

# Converting the nets to .nnue

The trainer outputs the networks in .ckpt format. The engines support only the .nnue format. To convert between the two one may run `python serialize.py --features=<feature set used for training> nn.ckpt nn.nnue`. This conversion is lossy as the .nnue format stores the weights in a quantized form, removes the training and optimizer state, and coalesces virtual feature weights into real feature weights.

# What to expect

Usually it takes about 400 epochs for the nets to mature, however they can be very competitive even after only 100 epochs. After 400 epochs the nets cannot improve much, and whether one ends up better than the other is mostly noise.

To compare different architectures/trainer settings one look at the validation loss between runs. But keep in mind that the validation loss depends on the dataset used and the way loss function is calculated. Even things like increasing fen skipping or disabling smart fen skipping affter the validation loss. In these cases it is necessary to play the games with the networks being produced.

There is some variance in the training. Usually when one trains multiple runs in the same way but with different random seeds (seeds are random if not specified) the resulting networks vary in strength measurably. It requires more than 1 run with the same settings to get a meaningful comparision when exploring code changes. There's a good chance that at least one run from a set of 4 is of relatively good quality. Rarely networks might even get corrupted during training and result in significantly worse playing strength while not showing any other symptoms (loss, visualization).

# Overseeing the training process through tensorboard

TODO: this