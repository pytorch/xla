# OpenXLA

PyTorch/XLA is a Python package that uses the [XLA deep learning
compiler](https://www.tensorflow.org/xla) to connect the [PyTorch deep learning
framework](https://pytorch.org/) and [Cloud
TPUs](https://cloud.google.com/tpu/). PyTorch/XLA now pull XLA from OpenXLA. 
OpenXLA here means [open source machine learning compiler XLA for GPUs, CPUs, and ML accelerators](https://github.com/openxla/xla), 
which belongs to the same-name community [OpenXLA Community](https://github.com/openxla) and this community also contain [StableHLO](https://github.com/openxla/stablehlo).

PyTorch/XLA used to pull XLA from [TensorFlow](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla) before https://github.com/pytorch/xla/pull/5202.
After https://github.com/pytorch/xla/pull/5202, PyTorch/XLA has migrated from pull XLA from [TensorFlow](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla)
to pull XLA from [OpenXLA](https://github.com/openxla/xla).


# How to use OpenXLA

For [PJRT runtime](https://github.com/pytorch/xla/blob/master/docs/pjrt.md) user,
there is no logic change before and after [Migrate PyTorch/XLA to pull XLA from OpenXLA](https://github.com/pytorch/xla/pull/5202).
For XRT runtime user, because OpenXLA don't support XRT, user could use XRT from [XRT branch of PyTorch/XLA](https://github.com/pytorch/xla/tree/xrt).


# Performance
||Training Throughput on ResNet50|Training Throughput on ResNet50 |Training Throughput on ResNet50 |
| :------------  | :------------  | :------------  | :------------  |
| model-pjrt-tpu  | resnet50-pjrt-v2-8 | resnet50-pjrt-v4-8 | resnet50-pjrt-v4-32 |
| Pre Migration  | 18.59    | 20.06 | 27.92 |
| Post Migration | 18.63    | 19.94 | 27.14 |


# pin dependency sunset
PyTorch/XLA pull XLA from OpenXLA via pin to a specific commit

# other `OpenXLA` or `openxla` used in PyTorch or PyTorch/XLA

