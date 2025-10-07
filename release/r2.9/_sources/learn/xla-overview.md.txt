# Pytorch/XLA Overview

PyTorch/XLA is an open-source Python package that enables PyTorch to run on XLA
(Accelerated Linear Algebra) compatible devices, with a primary focus on
**Google Cloud TPUs** and also supporting **XLA-compatible GPUs**. It allows
developers and researchers to leverage the massive parallel processing
capabilities of these accelerators for training and inferencing large-scale AI
models with minimal code changes from their existing PyTorch workflows.

At its core, PyTorch/XLA acts as a bridge between the familiar PyTorch Python
frontend and the XLA compiler. When you run PyTorch operations on XLA
devices using this library, you get the following key features:

1. **Lazy Evaluation**: Operations are not executed immediately. Instead,
   PyTorch/XLA records these operations in an intermediate representation (IR)
   graph. The process of generating the IR graph is often referred to as
   "tracing" (LazyTensor tracing or code tracing). Sometimes this is also called
   lazy evaluation and it can lead to significant
   [performance improvements](https://arxiv.org/pdf/2102.13267.pdf).
2. **Graph Compilation**: When results are actually needed (e.g., printing a
   tensor, saving a checkpoint, or at an explicit synchronization point like
   `torch_xla.sync()`), the accumulated IR graph is converted into a lower-level
   machine-readable format called HLO (High-Level Opcodes). HLO is a
   representation of a computation that is specific to the XLA compiler and
   allows it to generate efficient code for the hardware that it is running on.
3. **XLA Optimization**: The XLA compiler takes this HLO, performs a series of
   optimizations (like operator fusion, memory layout optimization, and
   parallelization), and compiles it into highly efficient machine code tailored
   for the specific XLA device (e.g., TPU).
4. **Execution**: The compiled code is then executed on the XLA device(s).
   Compiled graphs are cached, so subsequent executions with the same
   computation graph and input shapes can reuse the optimized binary,
   significantly speeding up repeated operations typical in training loops.

![img](../_static/img/pytorchXLA_flow.svg)

This process allows PyTorch/XLA to provide significant performance benefits,
especially for large models and distributed training scenarios. For a deeper
dive into the lazy tensor system, see our
[LazyTensor guide](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/).

## **Why Use PyTorch/XLA?**

* **High Performance on TPUs**: PyTorch/XLA is optimized to deliver exceptional performance for training and inference on Google Cloud TPUs, which are custom-designed AI accelerators.
* **Scalability**: Seamlessly scale your models from a single device to large TPU Pods with minimal code changes, enabling you to tackle more ambitious projects.
* **Familiar PyTorch Experience**: Continue using the PyTorch APIs and ecosystem you know and love. PyTorch/XLA aims to make the transition to XLA devices as smooth as possible, often requiring only minor modifications to existing PyTorch code.
* **Cost-Efficiency**: TPUs offer a compelling price/performance ratio for many AI workloads. PyTorch/XLA helps you take advantage of this efficiency.
* **Versatility**: Accelerate a wide range of AI workloads, including chatbots, code generation, media content generation, vision services, and recommendation engines.
* **Support for Leading Frameworks**: While focused on PyTorch, XLA itself is a compiler backend used by other major frameworks like JAX and TensorFlow.

## **Target Hardware**

While PyTorch/XLA can theoretically run on any XLA-compatible backend, its primary development and optimization focus is on:

* **Google Cloud TPUs**: Including various generations like TPU v5 and v6. [Learn more about TPUs](https://cloud.google.com/tpu/docs/intro-to-tpu).
* **GPUs via XLA**: PyTorch/XLA also supports running on NVIDIA GPUs through the OpenXLA PJRT plugin, providing an alternative execution path. [Learn more about GPUs on Google Cloud](https://cloud.google.com/compute/docs/gpus).

## TPU Setup

Create a TPU with the base image to use nightly wheels or from the stable
release by specifying the `RUNTIME_VERSION`.

``` bash
export ZONE=us-central2-b
export PROJECT_ID=your-project-id
export ACCELERATOR_TYPE=v4-8 # v4-16, v4-32, …
export RUNTIME_VERSION=tpu-vm-v4-pt-2.0 # or tpu-vm-v4-base
export TPU_NAME=your_tpu_name

gcloud compute tpus tpu-vm create ${TPU_NAME} \
--zone=${ZONE} \
--accelerator-type=${ACCELERATOR_TYPE} \
--version=${RUNTIME_VERSION} \
--subnetwork=tpusubnet
```

If you have a single host VM (e.g. v4-8), you can ssh to your vm and run
the following commands from the vm directly. Otherwise, in case of TPU
pods, you can use `--worker=all --command=""` similar to

``` bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
--zone=us-central2-b \
--worker=all \
--command="pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-nightly-cp38-cp38-linux_x86_64.whl"
```

Next, if you are using base image, install nightly packages and required
libraries

``` bash
pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-nightly-cp38-cp38-linux_x86_64.whl
  pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp38-cp38-linux_x86_64.whl
sudo apt-get install libopenblas-dev -y

sudo apt-get update && sudo apt-get install libgl1 -y # diffusion specific
```

## Next Steps

- [Examples](./xla-examples.md): Explore example code for training and inference on TPUs.
- [Profiling and Performance](./xla-profiling.md): Learn how to profile and optimize your PyTorch/XLA applications.
- [Advanced Topics](./xla-advanced.md): Dive deeper into advanced concepts like graph optimization, data loading, and distributed training with PyTorch/XLA.
