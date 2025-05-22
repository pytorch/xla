## Overview

This repo aims to provide some basic examples of how to run an existing PyTorch
model with PyTorch/XLA. `train_resnet_base.py` is a minimal trainer to run
ResNet50 with fake data on a single device. `train_decoder_only_base.py` is
similar to `train_resnet_base.py` but with a decoder-only model.

Other examples will import the `train_resnet_base` or `train_decoder_only_base`
and demonstrate how to enable different features (distributed training,
profiling, dynamo, etc.) on PyTorch/XLA. The objective of this repository is to
offer fundamental examples of executing an existing PyTorch model utilizing
PyTorch/XLA.

## Setup

Follow our [README](https://github.com/pytorch/xla#getting-started) to install
the latest release of torch_xla. Check out this
[link](https://github.com/pytorch/xla#python-packages) for torch_xla at other
versions. To install the nightly torchvision(required for the resnet) you can do

```shell
pip install --no-deps --pre torchvision -i https://download.pytorch.org/whl/nightly/cu118
```

## Run the example

You can run all models directly. The only environment variable you want to set
is `PJRT_DEVICE`.

```
PJRT_DEVICE=TPU python fsdp/train_decoder_only_fsdp_v2.py
```

## Examples and Description

- `train_resnet_base.py`: A minimal example of training ResNet50. This is the
  baseline example for comparing performance with other training strategies.

- `train_decoder_only_base.py`: A minimal example of training a decoder-only
  model. This serves as a baseline for comparison with other training
  strategies.

- `train_resnet_amp.py`: Shows how to use Automatic Mixed Precision (AMP) with
  PyTorch/XLA to improve performance. This example demonstrates the benefits of
  AMP for reducing memory usage and accelerating training.

- data_parallel: A trainer implementation to run ResNet50 on multiple devices
  using data-parallel.

  - `train_resnet_ddp.py`: Shows how to use PyTorch's DDP implementation for
    distributed training on TPUs. This example showcases how to integrate
    PyTorch's DDP with PyTorch/XLA for distributed training.
  - `train_resnet_spmd_data_parallel.py`: Leverages SPMD (Single Program
    Multiple Data) for distributed training. It shards the batch dimension
    across multiple devices and demonstrates how to achieve higher performance
    than DDP for specific workloads.
  - `train_resnet_xla_ddp.py`: Shows how to use PyTorch/XLA's built-in DDP
    implementation for distributed training on TPUs. It demonstrates the
    benefits of distributed training and the simplicity of using PyTorch/XLA's
    DDP.

- debug: A trainer implementation to run ResNet50 with debug mode.

  - `train_resnet_profile.py`: Captures performance insights with PyTorch/XLA's
    profiler to identify bottlenecks. Helps diagnose and optimize model
    performance.
  - `train_resnet_benchmark.py`: Provides a simple way to benchmark PyTorch/XLA,
    measuring device execution and tracing time for overall efficiency analysis.

- flash_attention: A trainer implementation to run a decoder-only model using
  Flash Attention.

  - `train_decoder_only_flash_attention.py`: Incorporates flash attention, an
    efficient attention mechanism, utilizing custom kernels for accelerated
    training.
  - `train_resnet_flash_attention_fsdp_v2.py`: Combines flash attention with
    FSDPv2, showcasing the integration of custom kernels with FSDP for scalable
    and efficient model training.

- fsdp: A trainer implementation to run a decoder-only model using FSDP (Fully
  Sharded Data Parallelism).

  - `train_decoder_only_fsdp_v2.py`: Employs FSDPv2(FSDP algorithm implemented
    with PyTorch/XLA GSPMD) for training the decoder-only model, demonstrating
    parallel training of large transformer models on TPUs.
  - `train_resnet_fsdp_auto_wrap.py`: Demonstrates FSDP (Fully Sharded Data
    Parallel) for model training, automatically wrapping model parts based on
    size or type criteria.
