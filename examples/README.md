## Overview
This repo aims to provide some basic examples of how to run an existing pytorch model with PyTorch/XLA. `train_resnet_base.py` is a minimal trainer to run ResNet50 with fake data on a single device. `train_decoder_only_base.py` is similar to `train_resnet_base.py` but with a decoder only model.

Other examples will import the `train_resnet_base` or `train_decoder_only_base` and demonstrate how to enable different features(distributed training, profiling, dynamo etc) on PyTorch/XLA.The objective of this repository is to offer fundamental examples of executing an existing PyTorch model utilizing PyTorch/XLA.

## Setup
Follow our [README](https://github.com/pytorch/xla#getting-started) to install latest release of torch_xla. Check out this [link](https://github.com/pytorch/xla#python-packages) for torch_xla at other versions. To install the nightly torchvision(required for the resnet) you can do

```shell
pip install --no-deps --pre torchvision -i https://download.pytorch.org/whl/nightly/cu118
```

## Run the example
You can run all models directly. Only environment you want to set is `PJRT_DEVICE`.
```
PJRT_DEVICE=TPU python fsdp/train_decoder_only_fsdp_v2.py
```
