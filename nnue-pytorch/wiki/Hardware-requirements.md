# Requirements

Minimal hardware requirements for running a single instance of the nnue-pytorch trainer:

- NVIDIA GPU (CUDA requirement)
- 4GB VRAM
- 2GB RAM

Reasonable hardware for running a single instance of the nnue-pytorch trainer:

- Nvidia GTX 1080ti
- 6GB VRAM
- 4GB RAM
- 8 CPU threads
- 20GB of disk space

Recommended hardware for running a single instance of the nnue-pytorch trainer:

- Nvidia RTX 2080
- 8GB VRAM
- 8GB RAM
- 12 CPU threads
- 50GB of disk space

Reasonable hardware for running network testing in real time alongside training:

- 400MB RAM / game concurrency
- 16 CPU threads

Recommended hardware for running network testing in real time alongside training:

- 500MB RAM / game concurrency
- 32 CPU threads

# Additional notes

It is possible to run multiple trainer instance in parallel. This is often done to fully saturate high-end GPUs and reduce impact of variance between runs. High-end GPUs usually require 2 runs in parallel on each GPU to saturate it.

Testing the networks as they are produced is a heavy task, and usually will impact training speed even when not utilizing virtual cores due to memory bandwidth.

# Benchmarks

TODO: A trainer benchmark script.