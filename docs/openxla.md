# OpenXLA

As of ***Jun 28***, PyTorch/XLA now ***pulls*** XLA from OpenXLA. 
OpenXLA ***is an*** [open source machine learning compiler XLA for GPUs, CPUs, and ML accelerators](https://github.com/openxla/xla). 

Previous to OpenXLA, PyTorch/XLA pulled XLA directly from [TensorFlow](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla). With our [XLA to OpenXLA migration](https://github.com/pytorch/xla/pull/5202), PyTorch/XLA now pulls XLA from [OpenXLA](https://github.com/openxla/xla).

# How to use OpenXLA

For [PJRT runtime](https://github.com/pytorch/xla/blob/master/docs/pjrt.md) users, there is no change with this migration. For XRT runtime users, there is a separate [XRT branch of PyTorch/XLA](https://github.com/pytorch/xla/tree/xrt) since OpenXLA doesn't support XRT, .


# Performance
||Training Throughput on ResNet50|Training Throughput on ResNet50 |Training Throughput on ResNet50 |
| :------------  | :------------  | :------------  | :------------  |
| model-pjrt-tpu  | resnet50-pjrt-v2-8 | resnet50-pjrt-v4-8 | resnet50-pjrt-v4-32 |
| Pre Migration  | 18.59    | 20.06 | 27.92 |
| Post Migration | 18.63    | 19.94 | 27.14 |


# pin dependency sunset
PyTorch/XLA pull XLA from OpenXLA via pin to a specific commit

# other `OpenXLA` or `openxla` used in PyTorch or PyTorch/XLA

