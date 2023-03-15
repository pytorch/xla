# PyTorch/XLA

<b>Current CI status:</b>  [![CircleCI](https://circleci.com/gh/pytorch/xla.svg?style=svg)](https://circleci.com/gh/pytorch/xla)

PyTorch/XLA is a Python package that uses the
[XLA deep learning compiler](https://www.tensorflow.org/xla)
to connect the [PyTorch deep learning framework](https://pytorch.org/) and
[Cloud TPUs](https://cloud.google.com/tpu/). You can try it right now, for free,
on a single Cloud TPU with [Google Colab](https://colab.research.google.com/),
and use it in production and on Cloud TPU Pods
with [Google Cloud](https://cloud.google.com/gcp).

Take a look at one of our Colab notebooks to quickly try different PyTorch networks
running on Cloud TPUs and learn how to use Cloud TPUs as PyTorch devices:

* [Getting Started with PyTorch on Cloud TPUs](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/getting-started.ipynb)
* [Training AlexNet on Fashion MNIST with a single Cloud TPU Core](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/single-core-alexnet-fashion-mnist.ipynb)
* [Training AlexNet on Fashion MNIST with multiple Cloud TPU Cores](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/multi-core-alexnet-fashion-mnist.ipynb)
* [Fast Neural Style Transfer (NeurIPS 2019 Demo)](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/style_transfer_inference.ipynb)
* [Training A Simple Convolutional Network on MNIST](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/mnist-training.ipynb)
* [Training a ResNet18 Network on CIFAR10](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/resnet18-training.ipynb)
* [ImageNet Inference with ResNet50](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/resnet50-inference.ipynb)
* [Training DC-GAN using Colab Cloud TPU](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/DC-GAN.ipynb)

The rest of this README covers:

* [User Guide & Best Practices](#user-guide--best-practices)
* [Running PyTorch on Cloud TPUs and GPU](#running-pytorchxla-on-cloud-tpu-and-gpu)
Google Cloud also runs networks faster than Google Colab.
* [Available docker images and wheels](#available-docker-images-and-wheels)
* [Performance Profiling and Auto-Metrics Analysis](#performance-profiling-and-auto-metrics-analysis)
* [Troubleshooting](#troubleshooting)
* [Providing Feedback](#providing-feedback)
* [Building and Contributing to PyTorch/XLA](#contributing)
* [Additional Reads](#additional-reads)



Additional information on PyTorch/XLA, including a description of its
semantics and functions, is available at [PyTorch.org](http://pytorch.org/xla/).

## User Guide & Best Practices

Our comprehensive user guides are available at:

[Documentation for the latest release](https://pytorch.org/xla)

[Documentation for master branch](https://pytorch.org/xla/master)

See the [API Guide](API_GUIDE.md) for best practices when writing networks that
run on XLA devices(TPU, GPU, CPU and...)

## Running PyTorch/XLA on Cloud TPU and GPU

* [Running on a single Cloud TPU](#running-on-a-single-cloud-tpu-vm)
* [Running on a Cloud TPU Pod](#how-to-run-on-tpu-vm-pods-distributed-training)
* [Running on a Cloud GPU](docs/gpu.md)

---

## Running on a Single Cloud TPU VM

Google Cloud offers TPU VMs for more transparent and easier access to the TPU hardware. This is our **recommended way** of running PyTorch/XLA on Cloud TPU. Please check out our [Cloud TPU VM User Guide](https://cloud.google.com/tpu/docs/pytorch-xla-ug-tpu-vm). To learn more about the Cloud TPU System Architecture, please check out [this doc](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu_vms).


---

## How to Run on TPU VM Pods (distributed training)

If a single TPU VM does not suit your requirment, you can consider using TPU Pod. TPU Pod is a collection of TPU devices connected by dedicated high-speed network interfaces. Please checkout our [Cloud TPU VM Pod User Guide](https://cloud.google.com/tpu/docs/pytorch-pods).


## Available docker images and wheels
### Docker
The following pre-built docker images are available. For running dockers, check [this doc](https://cloud.google.com/tpu/docs/pytorch-xla-ug-tpu-vm#docker-tpuvm) for TPUVM and [this doc](https://github.com/pytorch/xla/blob/master/docs/gpu.md#docker) for GPU.

| Version | Cloud TPU VMs Docker |
| --- | ----------- |
1.13 | `gcr.io/tpu-pytorch/xla:r1.13_3.8_tpuvm` |
1.12 | `gcr.io/tpu-pytorch/xla:r1.12_3.8_tpuvm` |
nightly | `gcr.io/tpu-pytorch/xla:nightly_3.8_tpuvm` |
nightly at date | `gcr.io/tpu-pytorch/xla:nightly_3.8_YYYYMMDD` |

<br/>

| Version | GPU CUDA 11.2 + Python 3.7 Docker |
| --- | ----------- |
1.13 | `gcr.io/tpu-pytorch/xla:r1.13_3.7_cuda_11.2` |
1.12 | `gcr.io/tpu-pytorch/xla:r1.12_3.7_cuda_11.2` |

<br/>

| Version | GPU CUDA 11.2 + Python 3.8 Docker |
| --- | ----------- |
| 1.13 | `gcr.io/tpu-pytorch/xla:r1.13_3.8_cuda_11.2` |

<br/>

| Version | GPU CUDA 11.7 + Python 3.8 Docker |
| --- | ----------- |
| nightly | `gcr.io/tpu-pytorch/xla:nightly_3.8_cuda_11.7` |
| nightly at date(>=20230210) | `gcr.io/tpu-pytorch/xla:nightly_3.8_cuda_11.7_YYYYMMDD` |

<br/>

| Version | GPU CUDA 11.8 + Python 3.8 Docker |
| --- | ----------- |
| nightly | `gcr.io/tpu-pytorch/xla:nightly_3.8_cuda_11.8` |
| nightly at date(>=20230210) | `gcr.io/tpu-pytorch/xla:nightly_3.8_cuda_11.8_YYYYMMDD` |

To run on [compute instances with GPUs](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus).

### Wheel
| Version | Cloud TPU VMs Wheel |
| --- | ----------- |
| 1.13 | `https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.13-cp38-cp38-linux_x86_64.whl` |
| 1.12 | `https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.12-cp38-cp38-linux_x86_64.whl` |
| 1.11 | `https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.11-cp38-cp38-linux_x86_64.whl` |
| 1.10 | `https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.10-cp38-cp38-linux_x86_64.whl` |
| 1.9 | `https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.9-cp38-cp38-linux_x86_64.whl` |
| nightly | `https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-nightly-cp38-cp38-linux_x86_64.whl` |

<br/>

| Version | GPU Wheel + Python 3.7 |
| --- | ----------- |
| 1.13 | `https://storage.googleapis.com/tpu-pytorch/wheels/cuda/112/torch_xla-1.13-cp37-cp37m-linux_x86_64.whl` |
| 1.12 | `https://storage.googleapis.com/tpu-pytorch/wheels/cuda/112/torch_xla-1.12-cp37-cp37m-linux_x86_64.whl` |
| 1.11 | `https://storage.googleapis.com/tpu-pytorch/wheels/cuda/112/torch_xla-1.11-cp37-cp37m-linux_x86_64.whl` |
| nightly | `https://storage.googleapis.com/tpu-pytorch/wheels/cuda/112/torch_xla-nightly-cp37-cp37-linux_x86_64.whl` |

<br/>

| Version | GPU Wheel + Python 3.8 |
| --- | ----------- |
| 1.13 | `https://storage.googleapis.com/tpu-pytorch/wheels/cuda/112/torch_xla-1.13-cp38-cp38-linux_x86_64.whl` |
| nightly + CUDA 11.7 | `https://storage.googleapis.com/tpu-pytorch/wheels/cuda/117/torch_xla-nightly-cp38-cp38-linux_x86_64.whl` |
| nightly + CUDA 11.8 | `https://storage.googleapis.com/tpu-pytorch/wheels/cuda/118/torch_xla-nightly-cp38-cp38-linux_x86_64.whl` |

<br/>

| Version | Colab TPU Wheel |
| --- | ----------- |
| 1.13 | `https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-1.13-cp37-cp37m-linux_x86_64.whl` |
| 1.12 | `https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-1.12-cp37-cp37m-linux_x86_64.whl` |
| 1.11 | `https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-1.11-cp37-cp37m-linux_x86_64.whl` |

You can also add `+yyyymmdd` after `torch_xla-nightly` to get the nightly wheel of a specified date. To get the companion pytorch and torchvision nightly wheel, replace the `torch_xla` with `torch` or `torchvision` on above wheel links.

### Installing libtpu

For PyTorch/XLA release r1.13 and older and when developing PyTorch/XLA, install the `libtpu` pip package with the following command:

```
pip3 install torch_xla[tpuvm]
```

This is only required on Cloud TPU VMs.

## Performance Profiling and Auto-Metrics Analysis

With PyTorch/XLA we provide a set of performance profiling tooling and auto-metrics analysis which you can check the following resources:
* [Official tutorial](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
* [Colab notebook](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/pytorch-xla-profiling-colab.ipynb)
* [Sample MNIST training script with profiling](https://github.com/pytorch/xla/blob/master/test/test_profile_mp_mnist.py)
* [Utility script for capturing performance profiles](https://github.com/pytorch/xla/blob/master/scripts/capture_profile.py)

## Troubleshooting

If PyTorch/XLA isn't performing as expected, see the
[troubleshooting guide](TROUBLESHOOTING.md), which has suggestions for
debugging and optimizing your network(s).

## Providing Feedback

The PyTorch/XLA team is always happy to hear from users and OSS contributors!
The best way to reach out is by filing an issue on this Github. Questions,
bug reports, feature requests, build issues, etc. are all welcome!

## Contributing

See the [contribution guide](CONTRIBUTING.md).

## Disclaimer
This repository is jointly operated and maintained by Google, Facebook and a number of individual contributors listed in the [CONTRIBUTORS](https://github.com/pytorch/xla/graphs/contributors) file. For questions directed at Facebook, please send an email to opensource@fb.com. For questions directed at Google, please send an email to pytorch-xla@googlegroups.com. For all other questions, please open up an issue in this repository [here](https://github.com/pytorch/xla/issues).

## Additional Reads
You can find additional useful reading materials in
* [Performance debugging on Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1)
* [Lazy tensor intro](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)
* [Scaling deep learning workloads with PyTorch / XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
* [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)
